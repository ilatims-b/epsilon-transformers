import argparse
import json
import os
import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm

from epsilon_transformers.persistence import Persister
from epsilon_transformers.process.processes import PROCESS_REGISTRY

# ----------------- Configuration & Load -----------------
parser = argparse.ArgumentParser(description="Analyze Sequence-Dependent KL Divergence for Mixed Processes")
parser.add_argument("-c", "--checkpoint_dir", type=str, required=True)
parser.add_argument("-m", "--max_branches", type=int, default=10000)
parser.add_argument("-o", "--output_dir", type=str, required=True)
args = parser.parse_args()

CHECKPOINT_DIR = args.checkpoint_dir
MAX_BRANCHES   = args.max_branches
OUTPUT_DIR     = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Device ----
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# ---- Load config ----
config_path = Path(CHECKPOINT_DIR) / "train_config.json"
with open(config_path) as f:
    train_config = json.load(f)

mixing_params    = train_config.get("dataset", {}).get("mixing_params", {})
processes_config = mixing_params.get("processes", [])
if len(processes_config) != 2:
    raise ValueError("KL script currently supports exactly 2 mixed processes.")

processes = []
for p_conf in processes_config:
    p_name     = p_conf[0]
    p_args     = p_conf[1]
    p_vocab_str = p_conf[2] if len(p_conf) > 2 else None
    p_vocab    = {int(k): v for k, v in p_vocab_str.items()} if p_vocab_str else None
    ProcessClass = PROCESS_REGISTRY[p_name]
    processes.append(ProcessClass(vocab_map=p_vocab, **p_args))

p0, p1 = processes
# Force float32 — required for MPS, good practice everywhere
T0 = torch.tensor(p0.transition_matrix, dtype=torch.float32, device=device)
T1 = torch.tensor(p1.transition_matrix, dtype=torch.float32, device=device)

switch_times = mixing_params.get("switch_times", [])
switch_probs = mixing_params.get("switch_prob", [])
if isinstance(switch_probs, float):
    switch_probs = [switch_probs] * len(switch_times)

D_VOCAB = T0.shape[0]   # emission dim
S0      = p0.num_states
S1      = p1.num_states

persister = Persister(Path(CHECKPOINT_DIR))
model = persister.load_final_model().to(device)
model.eval()

# ----------------- Helper Functions -----------------

def get_switch_prob(t: int) -> float:
    return switch_probs[switch_times.index(t)] if t in switch_times else 0.0


def get_next_beliefs_and_probs_batch(A_batch: torch.Tensor, switch_prob_t: float):
    """
    Fully vectorized batched belief update.

    Args:
        A_batch:       (N, 2, S0, S1)  – current joint beliefs for N branches
        switch_prob_t: scalar switch probability at this timestep

    Returns:
        A_next:  (N, E, 2, S0, S1)  – posterior belief for each branch × emission
        P_e:     (N, E)              – predictive prob of each emission per branch
    """
    # --- 1. Switch dynamics ---
    A_sw = torch.empty_like(A_batch)                                        # (N, 2, S0, S1)
    A_sw[:, 0] = A_batch[:, 0] * (1 - switch_prob_t) + A_batch[:, 1] * switch_prob_t
    A_sw[:, 1] = A_batch[:, 1] * (1 - switch_prob_t) + A_batch[:, 0] * switch_prob_t

    # --- 2. Transition (only active process moves) ---
    # unnorm_0[n,e,s0',s1] = Σ_s0  A_sw[n,0,s0,s1] * T0[e,s0,s0']
    unnorm_0 = torch.einsum('nij,eik->nekj', A_sw[:, 0], T0)               # (N, E, S0, S1)
    # unnorm_1[n,e,s0,s1'] = Σ_s1  A_sw[n,1,s0,s1] * T1[e,s1,s1']
    unnorm_1 = torch.einsum('nij,ejk->neik', A_sw[:, 1], T1)               # (N, E, S0, S1)

    # --- 3. Marginal emission probability ---
    P_e = unnorm_0.sum(dim=(2, 3)) + unnorm_1.sum(dim=(2, 3))              # (N, E)

    # --- 4. Bayesian update (safe division) ---
    safe_P = P_e.clamp(min=1e-30).unsqueeze(2).unsqueeze(3)                # (N, E, 1, 1)
    A_next_0 = unnorm_0 / safe_P                                            # (N, E, S0, S1)
    A_next_1 = unnorm_1 / safe_P                                            # (N, E, S0, S1)

    # Stack along process dim → (N, E, 2, S0, S1)
    A_next = torch.stack([A_next_0, A_next_1], dim=2)

    # Zero out numerically invalid entries
    invalid = (P_e < 1e-30).unsqueeze(2).unsqueeze(3).unsqueeze(4)         # (N, E, 1, 1, 1)
    A_next = A_next.masked_fill(invalid, 0.0)

    return A_next, P_e                                                       # (N,E,2,S0,S1), (N,E)


def model_probs_chunked(branch_seqs: list, t: int, chunk_size: int = 4096) -> torch.Tensor:
    """Run model on all branch sequences, return softmax probs (N, D_VOCAB)."""
    N = len(branch_seqs)
    tf_probs_list = []
    for i in range(0, N, chunk_size):
        chunk = branch_seqs[i : i + chunk_size]
        pad   = np.zeros((len(chunk), 100), dtype=np.int64)
        for j, s in enumerate(chunk):
            pad[j, :len(s)] = s
        x = torch.tensor(pad, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits[:, t - 1, :], dim=-1)
        tf_probs_list.append(probs)
    tf_probs = torch.cat(tf_probs_list, dim=0)          # (N, vocab_full)
    tf_probs = tf_probs[:, :D_VOCAB]                    # trim to process vocab
    tf_probs = tf_probs / tf_probs.sum(dim=1, keepdim=True).clamp(min=1e-30)
    return tf_probs                                      # (N, D_VOCAB)


# ----------------- Initialise State -----------------

steady0 = torch.tensor(p0.steady_state_vector, dtype=torch.float32, device=device)
steady1 = torch.tensor(p1.steady_state_vector, dtype=torch.float32, device=device)

A_init = torch.zeros(2, S0, S1, dtype=torch.float32, device=device)
A_init[0] = torch.outer(steady0, steady1)               # P0 active; P1 at steady state

# Tensors tracking N live branches
branch_beliefs = A_init.unsqueeze(0)                    # (1, 2, S0, S1)
branch_probs   = torch.ones(1, dtype=torch.float32, device=device)
branch_seqs    = [[]]                                   # list of token lists (CPU-side only)

# Output accumulators
avg_kl_pq       = torch.zeros(100, device=device)
avg_kl_qp       = torch.zeros(100, device=device)
std_kl_pq       = torch.zeros(100, device=device)
std_kl_qp       = torch.zeros(100, device=device)
kl_indep_pq     = torch.zeros(100, device=device)
kl_indep_qp     = torch.zeros(100, device=device)
P_marg_over_time = torch.zeros(100, D_VOCAB, device=device)
Q_marg_over_time = torch.zeros(100, D_VOCAB, device=device)

# ----------------- Main BFS Loop -----------------

for t in tqdm(range(100), desc="Searching Tree Sequences"):
    sw_prob = get_switch_prob(t)
    N = branch_beliefs.shape[0]

    # ---- Batched belief update (GPU) ----
    A_next_all, P_e_all = get_next_beliefs_and_probs_batch(branch_beliefs, sw_prob)
    # A_next_all: (N, E, 2, S0, S1)
    # P_e_all:    (N, E)

    # ---- KL computation (only from t=1 onwards; t=0 has no model prediction yet) ----
    if t > 0:
        # Model predictions
        tf_probs = model_probs_chunked(branch_seqs, t)  # (N, D_VOCAB)

        Q_hmm = P_e_all                                 # (N, D_VOCAB)  — HMM predictive dist

        # Normalised branch weights
        total_prob   = branch_probs.sum()
        norm_weights = branch_probs / total_prob.clamp(min=1e-30)   # (N,)

        # Per-branch KL(P_TF || Q_HMM) and KL(Q_HMM || P_TF)
        P_safe = tf_probs.clamp(1e-12, 1.0)
        Q_safe = Q_hmm.clamp(1e-12, 1.0)
        log_ratio_pq = torch.log(P_safe) - torch.log(Q_safe)       # (N, D_VOCAB)
        log_ratio_qp = -log_ratio_pq

        kl_pq = (tf_probs * log_ratio_pq).sum(dim=1)               # (N,)
        kl_qp = (Q_hmm   * log_ratio_qp).sum(dim=1)                # (N,)

        mean_pq = (norm_weights * kl_pq).sum()
        mean_qp = (norm_weights * kl_qp).sum()
        avg_kl_pq[t] = mean_pq
        avg_kl_qp[t] = mean_qp
        std_kl_pq[t] = torch.sqrt((norm_weights * (kl_pq - mean_pq).pow(2)).sum().clamp(min=0))
        std_kl_qp[t] = torch.sqrt((norm_weights * (kl_qp - mean_qp).pow(2)).sum().clamp(min=0))

        # Marginal distributions (weighted avg over branches)
        P_marg = (norm_weights.unsqueeze(1) * tf_probs).sum(dim=0)  # (D_VOCAB,)
        Q_marg = (norm_weights.unsqueeze(1) * Q_hmm  ).sum(dim=0)   # (D_VOCAB,)
        P_marg_over_time[t] = P_marg
        Q_marg_over_time[t] = Q_marg

        # Independent (marginal-level) KL
        Ps = P_marg.clamp(1e-12, 1.0);  Qs = Q_marg.clamp(1e-12, 1.0)
        kl_indep_pq[t] = (P_marg * (torch.log(Ps) - torch.log(Qs))).sum()
        kl_indep_qp[t] = (Q_marg * (torch.log(Qs) - torch.log(Ps))).sum()

        print(f"t={t:3d} | branches={N:6d} | beam_coverage={total_prob.item():.5f} "
              f"| KL(Q||P)={mean_qp.item():.5f} | KL(P||Q)={mean_pq.item():.5f}")

    # ---- Expand to next-level branches (vectorised) ----
    # new_probs[n, e] = branch_probs[n] * P_e_all[n, e]
    new_probs_full = branch_probs.unsqueeze(1) * P_e_all           # (N, E)

    # Flatten to (N*E,)
    new_probs_flat = new_probs_full.reshape(-1)                     # (N*E,)
    A_next_flat    = A_next_all.reshape(N * D_VOCAB, 2, S0, S1)    # (N*E, 2, S0, S1)
    valid_flat     = new_probs_flat > 1e-30                        # (N*E,)

    new_probs_valid = new_probs_flat[valid_flat]
    A_next_valid    = A_next_flat[valid_flat]

    # Reconstruct sequences (CPU, unavoidable for model forward pass)
    n_idx = torch.arange(N,       device=device).unsqueeze(1).expand(N, D_VOCAB).reshape(-1)
    e_idx = torch.arange(D_VOCAB, device=device).unsqueeze(0).expand(N, D_VOCAB).reshape(-1)
    n_valid_cpu = n_idx[valid_flat].cpu().tolist()
    e_valid_cpu = e_idx[valid_flat].cpu().tolist()
    new_seqs_valid = [branch_seqs[n] + [e] for n, e in zip(n_valid_cpu, e_valid_cpu)]

    # ---- Beam truncation (top-K by probability, GPU topk) ----
    n_valid = new_probs_valid.shape[0]
    if n_valid > MAX_BRANCHES:
        _, top_idx = torch.topk(new_probs_valid, MAX_BRANCHES)
        # Sort descending for consistency
        top_idx = top_idx[torch.argsort(new_probs_valid[top_idx], descending=True)]
        branch_probs   = new_probs_valid[top_idx]
        branch_beliefs = A_next_valid[top_idx]
        top_list       = top_idx.cpu().tolist()
        branch_seqs    = [new_seqs_valid[i] for i in top_list]
    else:
        branch_probs   = new_probs_valid
        branch_beliefs = A_next_valid
        branch_seqs    = new_seqs_valid

# ----------------- Visualisations -----------------

x_axis = np.arange(1, 100)

def to_np(t): return t.cpu().numpy()

avg_kl_qp_np  = to_np(avg_kl_qp)
std_kl_qp_np  = to_np(std_kl_qp)
avg_kl_pq_np  = to_np(avg_kl_pq)
std_kl_pq_np  = to_np(std_kl_pq)
kl_indep_qp_np = to_np(kl_indep_qp)
kl_indep_pq_np = to_np(kl_indep_pq)
P_marg_np     = to_np(P_marg_over_time)
Q_marg_np     = to_np(Q_marg_over_time)

upper_qp = avg_kl_qp_np[1:] + std_kl_qp_np[1:]
lower_qp = avg_kl_qp_np[1:] - std_kl_qp_np[1:]
upper_pq = avg_kl_pq_np[1:] + std_kl_pq_np[1:]
lower_pq = avg_kl_pq_np[1:] - std_kl_pq_np[1:]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=np.concatenate([x_axis, x_axis[::-1]]),
    y=np.concatenate([upper_qp, lower_qp[::-1]]),
    fill='toself', fillcolor='rgba(0,0,255,0.15)',
    line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
fig1.add_trace(go.Scatter(
    x=np.concatenate([x_axis, x_axis[::-1]]),
    y=np.concatenate([upper_pq, lower_pq[::-1]]),
    fill='toself', fillcolor='rgba(0,255,0,0.15)',
    line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
fig1.add_trace(go.Scatter(x=x_axis, y=avg_kl_qp_np[1:],
    name='Expected KL(Q || P) (Loss Measure)', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=x_axis, y=avg_kl_pq_np[1:],
    name='Expected KL(P || Q)', line=dict(color='green')))
for st in switch_times:
    if st > 0:
        fig1.add_vline(x=st, line_dash="dash", line_color="red", opacity=0.5)
fig1.update_layout(
    title="Sequence-Dependent KL Divergence over Time",
    xaxis_title="Predicted Token Index (Sequence Position)",
    yaxis_title="KL Divergence")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x_axis, y=kl_indep_qp_np[1:],
    name='Independent KL(Q || P)', line=dict(color='orange')))
fig2.update_layout(
    title="Sequence-Independent KL(Marginal Q_HMM || Marginal P_TF)",
    xaxis_title="Sequence Position",
    yaxis_title="KL Divergence (Bits)")

fig1.write_html(os.path.join(OUTPUT_DIR, "expected_kl_divergence.html"))
fig2.write_html(os.path.join(OUTPUT_DIR, "independent_kl_divergence.html"))
print(f"Plots saved to HTML in: {OUTPUT_DIR}")

try:
    fig1.write_image(os.path.join(OUTPUT_DIR, "expected_kl_divergence.png"))
    fig2.write_image(os.path.join(OUTPUT_DIR, "independent_kl_divergence.png"))
    print(f"PNG export successful.")
except Exception as e:
    print(f"Skipping PNG export (kaleido not installed?): {e}")

np.savez(
    os.path.join(OUTPUT_DIR, "kl_analysis_results.npz"),
    avg_kl_qp=avg_kl_qp_np, std_kl_qp=std_kl_qp_np,
    avg_kl_pq=avg_kl_pq_np, std_kl_pq=std_kl_pq_np,
    kl_indep_qp=kl_indep_qp_np, kl_indep_pq=kl_indep_pq_np,
    P_marg_over_time=P_marg_np, Q_marg_over_time=Q_marg_np,
)
print(f"Data saved to: {os.path.join(OUTPUT_DIR, 'kl_analysis_results.npz')}")
