import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import numpy as np

class MarkovKLAnalyzer:
    """Analyzes KL divergence against ground truth Markov process distributions."""
    
    def __init__(self, vocab_size: int, seq_len: Optional[int] = None):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    
    def _get_tensor(self, data, device):
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=torch.float32, device=device)
        return data.to(device=device, dtype=torch.float32)

    def _get_process_tensors(self, process, device):
        """Helper to get transition matrices for a single process."""
        T_emit = self._get_tensor(process.transition_matrix, device)
        if hasattr(process, 'norm_transition_matrix'):
            T_next = self._get_tensor(process.norm_transition_matrix, device)
        else:
            T_next = T_emit
        return T_emit, T_next

    def compute_ground_truth_distributions(self, 
                                          sequences: torch.Tensor, 
                                          process, 
                                          start_state_idx: Optional[int] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Computes the Ground Truth probability distributions for every step in the sequence.
        
        Args:
            sequences: [Batch, Seq_Len] tensor of token indices.
            process: The Process or MixedProcess object.
            start_state_idx: Optional starting state.
            
        Returns:
            all_gt_dists: [Batch, Seq_Len, Vocab_Size] Tensor of GT probabilities.
            all_kl_values: (Empty list in this function, kept for signature compatibility if needed)
        """
        if hasattr(process, 'processes') and hasattr(process, 'switch_schedule'):
            return self._compute_mixed_gt_distributions(sequences, process, start_state_idx)
        else:
            return self._compute_standard_gt_distributions(sequences, process, start_state_idx)

    def compute_kl_divergence_batch(self,
                                   model_logits: torch.Tensor,
                                   sequences: torch.Tensor,
                                   process,
                                   start_state_idx: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Get Ground Truth Distributions
        # gt_dists: [Batch, Seq_Len, Vocab]
        gt_dists = self.compute_ground_truth_distributions(sequences, process, start_state_idx)
        
        batch_size, seq_len, vocab_size = model_logits.shape
        device = model_logits.device
        
        kl_per_position = torch.zeros(seq_len, device=device)
        kl_all_values_list = []

        for pos in range(seq_len):
            # Model Logits
            logit_batch = model_logits[:, pos, :]
            model_log_probs = F.log_softmax(logit_batch, dim=-1)
            
            # GT Dist for this position
            gt_dist = gt_dists[:, pos, :]
            gt_log_probs = torch.log(gt_dist + 1e-10)
            
            # KL Calculation
            kl_batch = torch.sum(gt_dist * (gt_log_probs - model_log_probs), dim=-1)
            
            kl_per_position[pos] = kl_batch.mean()
            kl_all_values_list.append(kl_batch)

        if len(kl_all_values_list) > 0:
            kl_all_values = torch.cat(kl_all_values_list, dim=0)
        else:
            kl_all_values = torch.tensor([], device=device)

        return kl_per_position, kl_all_values

    def _compute_standard_gt_distributions(self, sequences, process, start_state_idx):
        batch_size, seq_len = sequences.shape
        device = sequences.device
        T_emit, T_next = self._get_process_tensors(process, device)
        
        # Initialize Belief
        if start_state_idx is not None:
            vec = torch.zeros(process.num_states, device=device, dtype=torch.float32)
            vec[start_state_idx] = 1.0
        else:
            vec = process.steady_state_vector
            if isinstance(vec, np.ndarray):
                vec = torch.from_numpy(vec).float().to(device)
        current_states = vec.unsqueeze(0).expand(batch_size, -1)
        
        T_emit_marginal = T_emit.sum(dim=2).t()
        
        all_gt_dists = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)

        for pos in range(seq_len):
            # 1. P(Next Token)
            gt_dist = torch.matmul(current_states, T_emit_marginal)
            
            # Map to global vocab
            if process.vocab_map is not None:
                gt_dist_mapped = torch.zeros(batch_size, self.vocab_size, device=device)
                if isinstance(process.vocab_map, dict):
                    vals = list(process.vocab_map.values())
                    keys = list(process.vocab_map.keys())
                    indices = torch.tensor(vals, device=device, dtype=torch.long)
                    gt_dist_mapped.index_add_(1, indices, gt_dist[:, keys])
                elif isinstance(process.vocab_map, list):
                    indices = torch.tensor(process.vocab_map, device=device, dtype=torch.long)
                    gt_dist_mapped.index_add_(1, indices, gt_dist)
                gt_dist = gt_dist_mapped

            gt_dist = gt_dist / (gt_dist.sum(dim=1, keepdim=True) + 1e-12)
            all_gt_dists[:, pos, :] = gt_dist

            # 2. Update Belief
            # Inverse Map: We need to find local token index.
            global_emissions = sequences[:, pos]
            local_emissions = global_emissions.clone()
            
            # Simple 1-to-1 inverse map assumption for standard process
            if process.vocab_map is not None:
                # Build lookup table dynamically or precompute
                lookup = torch.full((self.vocab_size,), -1, device=device, dtype=torch.long)
                if isinstance(process.vocab_map, dict):
                    for k, v in process.vocab_map.items(): lookup[v] = k
                else:
                    for k, v in enumerate(process.vocab_map): lookup[v] = k
                local_emissions = lookup[global_emissions]

            T_selected = T_next[local_emissions]
            next_states = torch.einsum("bs, bsd -> bd", current_states, T_selected)
            current_states = next_states / (next_states.sum(dim=1, keepdim=True) + 1e-12)

        return all_gt_dists

    def _compute_mixed_gt_distributions(self, sequences, mixed_process, start_state_idx):
        batch_size, seq_len = sequences.shape
        device = sequences.device
        num_procs = mixed_process.num_processes
        
        # --- 1. Precompute ---
        process_data = []
        for p in mixed_process.processes:
            T_emit, T_next = self._get_process_tensors(p, device)
            T_emit_marginal = T_emit.sum(dim=2).t()
            
            v_map_indices = None
            if p.vocab_map is not None:
                if isinstance(p.vocab_map, dict):
                    v_map_indices = torch.tensor(list(p.vocab_map.values()), device=device, dtype=torch.long)
                else:
                    v_map_indices = torch.tensor(p.vocab_map, device=device, dtype=torch.long)
            
            inverse_lookup = torch.full((self.vocab_size,), -1, device=device, dtype=torch.long)
            if p.vocab_map is not None:
                if isinstance(p.vocab_map, dict):
                    for k, v in p.vocab_map.items(): inverse_lookup[v] = k
                else:
                    for k, v in enumerate(p.vocab_map): inverse_lookup[v] = k

            process_data.append({
                'T_emit_marginal': T_emit_marginal,
                'T_next': T_next,
                'v_map_indices': v_map_indices,
                'inverse_lookup': inverse_lookup
            })

        # --- 2. Initialize ---
        active_proc_probs = torch.zeros(batch_size, num_procs, device=device)
        active_proc_probs[:, 0] = 1.0 

        # Handle switching at pos=0 if needed (affects prior before we see x_0, 
        # but we are skipping prediction of x_0 anyway, so this sets up belief for after x_0)
        # However, observe_update will fix belief based on x_0. 
        # The switch_schedule check later handles switches between x_t and x_{t+1}.

        proc_beliefs = []
        for p in mixed_process.processes:
            if start_state_idx is not None:
                vec = torch.zeros(p.num_states, device=device)
                vec[start_state_idx] = 1.0
            else:
                vec = p._get_gpu_steady_state(device)
            proc_beliefs.append(vec.unsqueeze(0).expand(batch_size, -1))

        all_gt_dists = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)

        # --- 3. Sequence Loop ---
        for pos in range(seq_len):
            
            # --- Step A: OBSERVE x_t & UPDATE POSTERIOR ---
            # We see token x_t. This updates our knowledge of the CURRENT state.
            global_token = sequences[:, pos]
            
            posterior_proc_probs = torch.zeros_like(active_proc_probs)
            evolved_beliefs = [b.clone() for b in proc_beliefs]
            
            for k in range(num_procs):
                lookup = process_data[k]['inverse_lookup']
                local_tokens = lookup[global_token] 
                valid_mask = (local_tokens != -1)
                
                # 1. Likelihood P(x_t | k)
                p_x_given_k = torch.zeros(batch_size, device=device)
                if valid_mask.any():
                    cols = process_data[k]['T_emit_marginal'][:, local_tokens[valid_mask]]
                    p_val = (proc_beliefs[k][valid_mask] * cols.t()).sum(dim=1)
                    p_x_given_k[valid_mask] = p_val

                # 2. Posterior Unnormalized
                posterior_proc_probs[:, k] = p_x_given_k * active_proc_probs[:, k]
                
                # 3. Evolve Belief State (Filter)
                if valid_mask.any():
                    T_next = process_data[k]['T_next']
                    T_sel = T_next[local_tokens[valid_mask]]
                    next_b = torch.einsum("bs, bsd -> bd", proc_beliefs[k][valid_mask], T_sel)
                    evolved_beliefs[k][valid_mask] = next_b / (next_b.sum(dim=1, keepdim=True) + 1e-12)

            # Normalize Posteriors (Collapse to one-hot if disjoint)
            posterior_sum = posterior_proc_probs.sum(dim=1, keepdim=True)
            clean_posteriors = posterior_proc_probs / (posterior_sum + 1e-12)

            # --- Step B: STATE MODE LOGIC ---
            if mixed_process.state_mode == 'same':
                avg_belief = torch.zeros_like(proc_beliefs[0])
                for k in range(num_procs):
                    w = clean_posteriors[:, k].unsqueeze(1)
                    avg_belief += evolved_beliefs[k] * w
                avg_belief = avg_belief / (avg_belief.sum(dim=1, keepdim=True) + 1e-12)
                for k in range(num_procs):
                    proc_beliefs[k] = avg_belief.clone()
            
            elif mixed_process.state_mode == 'resume':
                for k in range(num_procs):
                    w = clean_posteriors[:, k].unsqueeze(1)
                    proc_beliefs[k] = w * evolved_beliefs[k] + (1 - w) * proc_beliefs[k]
                    
            elif mixed_process.state_mode == 'steady':
                for k in range(num_procs):
                    proc_beliefs[k] = evolved_beliefs[k]

            active_proc_probs = clean_posteriors

            # --- Step C: SWITCHING LOGIC (Prior for NEXT token x_{t+1}) ---
            # Will a switch happen between index `pos` and `pos+1`?
            next_pos = pos + 1
            if next_pos in mixed_process.switch_schedule:
                p_switch = mixed_process.switch_schedule[next_pos]
                
                new_active_probs = torch.zeros_like(active_proc_probs)
                new_beliefs = [] 

                for k in range(num_procs):
                    prev_k = (k - 1) % num_procs
                    
                    prob_stay = (1 - p_switch) * active_proc_probs[:, k]
                    prob_arrive = p_switch * active_proc_probs[:, prev_k]
                    total_prob = prob_stay + prob_arrive
                    new_active_probs[:, k] = total_prob

                    if mixed_process.state_mode == 'steady':
                        steady_vec = mixed_process.processes[k]._get_gpu_steady_state(device)
                        steady_vec = steady_vec.unsqueeze(0).expand(batch_size, -1)
                        w_stay = (prob_stay / (total_prob + 1e-12)).unsqueeze(1)
                        w_arrive = (prob_arrive / (total_prob + 1e-12)).unsqueeze(1)
                        mixed_belief = w_stay * proc_beliefs[k] + w_arrive * steady_vec
                        new_beliefs.append(mixed_belief)
                
                active_proc_probs = new_active_probs
                if mixed_process.state_mode == 'steady':
                    proc_beliefs = new_beliefs

            # --- Step D: PREDICT NEXT TOKEN x_{t+1} ---
            gt_dist_total = torch.zeros(batch_size, self.vocab_size, device=device)
            
            for k in range(num_procs):
                local_dist = torch.matmul(proc_beliefs[k], process_data[k]['T_emit_marginal'])
                mapped_dist = torch.zeros(batch_size, self.vocab_size, device=device)
                indices = process_data[k]['v_map_indices']
                
                if indices is not None:
                    mapped_dist.index_add_(1, indices, local_dist)
                else:
                    mapped_dist = local_dist
                
                weight = active_proc_probs[:, k].unsqueeze(1)
                gt_dist_total += mapped_dist * weight

            gt_dist_total = gt_dist_total / (gt_dist_total.sum(dim=1, keepdim=True) + 1e-12)
            
            # Store at `pos`, aligning with logits[:, pos] which predicts x_{t+1}
            all_gt_dists[:, pos, :] = gt_dist_total

        return all_gt_dists 

def compute_markov_kl_divergence(model_logits, sequences, process, analyzer=None, return_per_position=True, start_state_idx: Optional[int]=None) -> Dict[str, float]:
    if analyzer is None:
        analyzer = MarkovKLAnalyzer(model_logits.shape[-1], model_logits.shape[1])
    
    kl_per_position, kl_all_values = analyzer.compute_kl_divergence_batch(
        model_logits, sequences, process, start_state_idx=start_state_idx
    )
    
    results = {
        "kl_div_markov": float(kl_all_values.mean().item()),
    }
    
    if return_per_position:
        kl_per_pos_cpu = kl_per_position.cpu().tolist()
        for pos_idx, val in enumerate(kl_per_pos_cpu):
            results[f"kl_div_markov_pos_{pos_idx + 1}"] = val
    
    return results