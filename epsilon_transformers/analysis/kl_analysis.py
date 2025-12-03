import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
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
    
    def compute_kl_divergence_batch(self,
                                   model_logits: torch.Tensor,
                                   sequences: torch.Tensor,
                                   process) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence by tracking state through sequence.
        Supports standard processes and Linear_Mess3 (NormTransitionMixin).
        """
        batch_size, seq_len, vocab_size = model_logits.shape
        device = model_logits.device
        
        # 1. Get Transition Matrices
        # T_emit: Used to compute P(Emission | Current State)
        # T_next: Used to compute P(Next State | Current State, Emission)
        
        T_emit = self._get_tensor(process.transition_matrix, device)
        
        # Check if process has a special normalization matrix (Linear_Mess3)
        if hasattr(process, 'norm_transition_matrix'):
            T_next = self._get_tensor(process.norm_transition_matrix, device)
        else:
            T_next = T_emit # Standard process uses same matrix for both

        vec = process.steady_state_vector
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec).float().to(device)
        
        current_states = vec.unsqueeze(0).expand(batch_size, -1)
        
        kl_per_position = torch.zeros(seq_len - 1, device=device)
        kl_all_values_list = []

        T_emit_marginal = T_emit.sum(dim=2).t() 

        for pos in range(1, seq_len):
            # P(e) = P(s) * P(e|s)
            # [B, S] @ [S, V] -> [B, V]
            emission_probs = torch.matmul(current_states, T_emit_marginal)
            
            emission_probs = emission_probs / (emission_probs.sum(dim=1, keepdim=True))
            gt_dist = emission_probs 

            #Model Probs
            logit_batch = model_logits[:, pos, :]
            model_log_probs = F.log_softmax(logit_batch, dim=-1)
            
            #KL 
            gt_log_probs = torch.log(gt_dist)
            kl_batch = torch.sum(gt_dist * (gt_log_probs - model_log_probs), dim=-1)
            
            kl_per_position[pos-1] = kl_batch.mean()
            kl_all_values_list.append(kl_batch)

            # Use T_next (which might be norm_transition_matrix)
            emissions = sequences[:, pos-1] # [B]
            
            # Select matrices for observed emissions: [B, S, D]
            T_selected = T_next[emissions]
            
            # Next state = Current State * Transition
            next_states = torch.einsum("bs, bsd -> bd", current_states, T_selected)
            
            # Normalize (if it is linear mess3, you are just normalizing by 1)
            next_states = next_states / (next_states.sum(dim=1, keepdim=True))
            current_states = next_states
        
        if len(kl_all_values_list) > 0:
            kl_all_values = torch.cat(kl_all_values_list, dim=0)
        else:
            kl_all_values = torch.tensor([], device=device)

        return kl_per_position, kl_all_values

def compute_markov_kl_divergence(model_logits, sequences, process, analyzer=None, return_per_position=True):
    if analyzer is None:
        analyzer = MarkovKLAnalyzer(model_logits.shape[-1], model_logits.shape[1])
    
    kl_per_position, kl_all_values = analyzer.compute_kl_divergence_batch(
        model_logits, sequences, process
    )
    
    results = {
        "kl_div_markov": float(kl_all_values.mean().item()),
    }
    
    if return_per_position:
        kl_per_pos_cpu = kl_per_position.cpu().tolist()
        for pos_idx, val in enumerate(kl_per_pos_cpu):
            results[f"kl_div_markov_pos_{pos_idx + 1}"] = val
    
    return results