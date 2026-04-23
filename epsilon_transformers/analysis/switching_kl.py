import argparse
import json
import os
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm

from epsilon_transformers.persistence import Persister
from epsilon_transformers.process.processes import PROCESS_REGISTRY

# ----------------- Configuration & Load -----------------
parser = argparse.ArgumentParser(description="Analyze Sequence-Dependent KL Divergence for Mixed Processes")
parser.add_argument("-c", "--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory containing train_config.json")
parser.add_argument("-m", "--max_branches", type=int, default=1000, help="Maximum number of branches to keep during beam search")
parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the plots")
args = parser.parse_args()

CHECKPOINT_DIR = args.checkpoint_dir
MAX_BRANCHES = args.max_branches
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load configuration dynamically
config_path = Path(CHECKPOINT_DIR) / "train_config.json"
with open(config_path, "r") as f:
    train_config = json.load(f)

mixing_params = train_config.get("dataset", {}).get("mixing_params", {})
processes_config = mixing_params.get("processes", [])

if len(processes_config) != 2:
    raise ValueError("KL script currently supports exactly 2 mixed processes.")

processes = []
for p_conf in processes_config:
    p_name = p_conf[0]
    p_args = p_conf[1]
    p_vocab_str = p_conf[2] if len(p_conf) > 2 else None
    
    p_vocab = {int(k): v for k, v in p_vocab_str.items()} if p_vocab_str else None
    ProcessClass = PROCESS_REGISTRY[p_name]
    p = ProcessClass(vocab_map=p_vocab, **p_args) 
    processes.append(p)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

p0, p1 = processes
T0 = torch.tensor(p0.transition_matrix, dtype=torch.float32, device=device)
T1 = torch.tensor(p1.transition_matrix, dtype=torch.float32, device=device)

switch_times = mixing_params.get("switch_times", [])
switch_probs = mixing_params.get("switch_prob", [])
if isinstance(switch_probs, float):
    switch_probs = [switch_probs] * len(switch_times)

D_VOCAB = T0.shape[0]

persister = Persister(Path(CHECKPOINT_DIR))
model = persister.load_final_model().to(device)
model.eval()

def get_switch_prob(t):
    if t in switch_times:
        return switch_probs[switch_times.index(t)]
    return 0.0

def compute_kl(P, Q):
    """Computes KL divergence robustly"""
    P_safe = torch.clamp(P, 1e-12, 1.0)
    Q_safe = torch.clamp(Q, 1e-12, 1.0)
    return torch.sum(P * torch.log(P_safe / Q_safe))

def get_next_beliefs_and_probs(A, switch_prob_t):
    """
    Given the current belief A[active_process, state0, state1], computes the next
    emission probabilities Q(x_{t+1}) and updates the belief given an emission.
    """
    # 1. Apply Switch Dynamics: 0 switches to 1, 1 switches to 0
    A_sw = torch.empty_like(A)
    A_sw[0] = A[0] * (1 - switch_prob_t) + A[1] * switch_prob_t
    A_sw[1] = A[1] * (1 - switch_prob_t) + A[0] * switch_prob_t
    
    # 2. Compute unnormalized transition joint paths
    unnorm_A_next_0 = torch.einsum('ij, eik -> ekj', A_sw[0], T0) # (e, s0_new, s1)
    unnorm_A_next_1 = torch.einsum('ij, ejk -> eik', A_sw[1], T1) # (e, s0, s1_new)
    
    # 3. Sum out state dimensions to get strict probability over emissions
    P_e = torch.sum(unnorm_A_next_0, dim=(1, 2)) + torch.sum(unnorm_A_next_1, dim=(1, 2))
    
    # 4. Normalize to obtain the Bayesian update conditional distributions
    A_next = torch.zeros((D_VOCAB, 2, A.shape[1], A.shape[2]), device=device)
    valid_e = P_e > 0
    A_next[valid_e, 0] = unnorm_A_next_0[valid_e] / P_e[valid_e, None, None]
    A_next[valid_e, 1] = unnorm_A_next_1[valid_e] / P_e[valid_e, None, None]
    
    return A_next, P_e

# ----------------- Execute BFS Tree Pruning -----------------

A_init = torch.zeros((2, p0.num_states, p1.num_states), device=device)
steady0 = torch.tensor(p0.steady_state_vector, dtype=torch.float32, device=device)
steady1 = torch.tensor(p1.steady_state_vector, dtype=torch.float32, device=device)
for i in range(p0.num_states):
    for j in range(p1.num_states):
        A_init[0, i, j] = steady0[i] * steady1[j]
        
current_branches = [{'seq': [], 'prob': 1.0, 'belief': A_init}]

avg_kl_pq = torch.zeros(100, device=device)
avg_kl_qp = torch.zeros(100, device=device)
std_kl_pq = torch.zeros(100, device=device)
std_kl_qp = torch.zeros(100, device=device)
kl_indep_pq = torch.zeros(100, device=device)
kl_indep_qp = torch.zeros(100, device=device)
P_marg_over_time = torch.zeros((100, D_VOCAB), device=device)
Q_marg_over_time = torch.zeros((100, D_VOCAB), device=device)

for t in tqdm(range(100), desc="Searching Tree Sequences"):
    sw_prob = get_switch_prob(t)
    next_branches = []
    
    if t > 0:
        N = len(current_branches)
        chunk_size = 4096
        tf_probs_all = []
        
        for i in range(0, N, chunk_size):
            chunk = current_branches[i:i+chunk_size]
            seqs = [b['seq'] for b in chunk]
            
            pad_seqs = np.zeros((len(chunk), 100), dtype=np.int64)
            for j, s in enumerate(seqs):
                pad_seqs[j, :len(s)] = s
                
            x_tensor = torch.tensor(pad_seqs, dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x_tensor) 
                probs = torch.softmax(logits[:, t-1, :], dim=-1)
            tf_probs_all.append(probs)
            
        tf_probs = torch.cat(tf_probs_all, dim=0) # [N, 6]
        
        kl_divs_pq = []
        kl_divs_qp = []
        branch_probs = []
        P_marg = torch.zeros(D_VOCAB, device=device)
        Q_marg = torch.zeros(D_VOCAB, device=device)
        total_prob = 0.0
        
        for i, branch in enumerate(current_branches):
            A_next, P_Hmm_e = get_next_beliefs_and_probs(branch['belief'], sw_prob)
            
            P_tf = tf_probs[i, :D_VOCAB]
            P_tf = P_tf / torch.sum(P_tf)
            Q_hmm = P_Hmm_e
            
            kl_pq = compute_kl(P_tf, Q_hmm)
            kl_qp = compute_kl(Q_hmm, P_tf)
            b_prob = branch['prob']
            
            kl_divs_pq.append(kl_pq)
            kl_divs_qp.append(kl_qp)
            branch_probs.append(b_prob)
            
            P_marg += b_prob * P_tf
            Q_marg += b_prob * Q_hmm
            total_prob += b_prob
            
            for e in range(D_VOCAB):
                p_val = Q_hmm[e].item()
                if p_val > 0:
                    new_prob = b_prob * p_val
                    next_branches.append({
                        'seq': branch['seq'] + [e],
                        'prob': new_prob,
                        'belief': A_next[e]
                    })
                        
        norm_b_probs = torch.tensor(branch_probs, dtype=torch.float32, device=device) / total_prob
        kl_divs_pq_tensor = torch.stack(kl_divs_pq)
        kl_divs_qp_tensor = torch.stack(kl_divs_qp)
        
        mean_pq = torch.sum(norm_b_probs * kl_divs_pq_tensor)
        mean_qp = torch.sum(norm_b_probs * kl_divs_qp_tensor)
        avg_kl_pq[t] = mean_pq
        avg_kl_qp[t] = mean_qp
        
        std_kl_pq[t] = torch.sqrt(torch.sum(norm_b_probs * (kl_divs_pq_tensor - mean_pq)**2))
        std_kl_qp[t] = torch.sqrt(torch.sum(norm_b_probs * (kl_divs_qp_tensor - mean_qp)**2))
        
        P_marg /= total_prob
        Q_marg /= total_prob
        
        print(f"Total probability at time {t}: {total_prob}")
        P_marg_over_time[t] = P_marg
        Q_marg_over_time[t] = Q_marg
        
        kl_indep_pq[t] = compute_kl(P_marg, Q_marg)
        kl_indep_qp[t] = compute_kl(Q_marg, P_marg)
        
    else: 
        for branch in current_branches:
            A_next, P_hmm_e = get_next_beliefs_and_probs(branch['belief'], sw_prob)
            for e in range(D_VOCAB):
                p_val = P_hmm_e[e].item()
                if p_val > 0:
                    new_prob = branch['prob'] * p_val
                    next_branches.append({
                        'seq': branch['seq'] + [e],
                        'prob': new_prob,
                        'belief': A_next[e]
                    })
                    
    # --- BEAM SEARCH TRUNCATION ---
    next_branches.sort(key=lambda x: x['prob'], reverse=True)
    current_branches = next_branches[:MAX_BRANCHES]

# ----------------- Visualizations -----------------

x_axis = np.arange(1, 100)

avg_kl_qp_np = avg_kl_qp.cpu().numpy()
std_kl_qp_np = std_kl_qp.cpu().numpy()
avg_kl_pq_np = avg_kl_pq.cpu().numpy()
std_kl_pq_np = std_kl_pq.cpu().numpy()
kl_indep_qp_np = kl_indep_qp.cpu().numpy()
kl_indep_pq_np = kl_indep_pq.cpu().numpy()
P_marg_over_time_np = P_marg_over_time.cpu().numpy()
Q_marg_over_time_np = Q_marg_over_time.cpu().numpy()

upper_qp = avg_kl_qp_np[1:] + std_kl_qp_np[1:]
lower_qp = avg_kl_qp_np[1:] - std_kl_qp_np[1:]
upper_pq = avg_kl_pq_np[1:] + std_kl_pq_np[1:]
lower_pq = avg_kl_pq_np[1:] - std_kl_pq_np[1:]

fig1 = go.Figure()

# Add shaded std error bands first so they sit behind the lines
fig1.add_trace(go.Scatter(x=np.concatenate([x_axis, x_axis[::-1]]), 
                          y=np.concatenate([upper_qp, lower_qp[::-1]]),
                          fill='toself', fillcolor='rgba(0,0,255,0.15)',
                          line=dict(color='rgba(255,255,255,0)'),
                          hoverinfo="skip",
                          showlegend=False))
                          
fig1.add_trace(go.Scatter(x=np.concatenate([x_axis, x_axis[::-1]]), 
                          y=np.concatenate([upper_pq, lower_pq[::-1]]),
                          fill='toself', fillcolor='rgba(0,255,0,0.15)',
                          line=dict(color='rgba(255,255,255,0)'),
                          hoverinfo="skip",
                          showlegend=False))

fig1.add_trace(go.Scatter(x=x_axis, y=avg_kl_qp_np[1:], name='Expected KL(Q || P) (Loss Measure)', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=x_axis, y=avg_kl_pq_np[1:], name='Expected KL(P || Q)', line=dict(color='green')))

for st in switch_times:
    if st > 0:
        fig1.add_vline(x=st, line_dash="dash", line_color="red", opacity=0.5)

fig1.update_layout(title="Sequence-Dependent KL Divergence over Time",
                   xaxis_title="Predicted Token Index (Sequence Position)",
                   yaxis_title="KL Divergence")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x_axis, y=kl_indep_qp_np[1:], name='Independent KL(Q || P)', line=dict(color='orange')))
fig2.update_layout(title="Sequence-Independent KL(Marginal Q_HMM || Marginal P_TF)",
                   xaxis_title="Sequence Position",
                   yaxis_title="KL Divergence (Bits)")

# Save logic
html_path_1 = os.path.join(OUTPUT_DIR, "expected_kl_divergence.html")
html_path_2 = os.path.join(OUTPUT_DIR, "independent_kl_divergence.html")

fig1.write_html(html_path_1)
fig2.write_html(html_path_2)

print(f"Plots saved to HTML in: {OUTPUT_DIR}")

# Optionally try to save as PNG (requires kaleido to be installed)
try:
    fig1.write_image(os.path.join(OUTPUT_DIR, "expected_kl_divergence.png"))
    fig2.write_image(os.path.join(OUTPUT_DIR, "independent_kl_divergence.png"))
    print(f"Plots successfully exported to PNG as well in {OUTPUT_DIR}")
except Exception as e:
    print(f"Skipping PNG export (is 'kaleido' installed?). Reason: {e}")

np.savez(
    os.path.join(OUTPUT_DIR, "kl_analysis_results.npz"),
    avg_kl_qp=avg_kl_qp_np,
    std_kl_qp=std_kl_qp_np,
    avg_kl_pq=avg_kl_pq_np,
    std_kl_pq=std_kl_pq_np,
    kl_indep_qp=kl_indep_qp_np,
    kl_indep_pq=kl_indep_pq_np,
    P_marg_over_time=P_marg_over_time_np,
    Q_marg_over_time=Q_marg_over_time_np
)
print(f"Data arrays cleanly exported to: {os.path.join(OUTPUT_DIR, 'kl_analysis_results.npz')}")
 