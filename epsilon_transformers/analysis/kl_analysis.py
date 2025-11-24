import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class MarkovKLAnalyzer:
    """Analyzes KL divergence against ground truth Markov process distributions."""
    
    def __init__(self, vocab_size: int, seq_len: Optional[int] = None):
        """
        Initialize the Markov KL analyzer.
        
        Args:
            vocab_size: Size of the vocabulary
            seq_len: Optional expected sequence length
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    
    # def _compute_emission_probabilities(self,
    #                                    process,
    #                                    state_prob_vector, xp) -> object:
    #     """
    #     Compute emission probabilities for the current state.
        
    #     This uses the internal _compute_emission_probabilities function from Process.
        
    #     Args:
    #         process: Process object with transition_matrix
    #         state_prob_vector: Current state probability vector (num_states,)
            
    #     Returns:
    #         Emission probability tensor of shape (vocab_size,)
    #     """
        
    #     try:
    #         # Import the helper function from process module
    #         from epsilon_transformers.process.Process import _compute_emission_probabilities
            
    #         # Compute emission probs given current state
    #         emission_probs = _compute_emission_probabilities(process, state_prob_vector)
    #         return emission_probs
    #     except ImportError:
    #         # Fallback: compute directly using the formula
    #         T = process.transition_matrix  # (vocab_len, num_states, num_states)
    #         emission_probs = np.einsum("s,esd->ed", state_prob_vector, T).sum(axis=1)
    #         emission_probs /= emission_probs.sum()
    #         return emission_probs
    def _compute_emission_probabilities(self, T, state_prob_vector, xp):
        emission_probs = xp.einsum("s,esd->ed", state_prob_vector, T).sum(axis=1)
        ep_sum = emission_probs.sum()
        if ep_sum != 0:
            emission_probs /= ep_sum
            
        return emission_probs
    # def _compute_next_state(self,
    #                        process,
    #                        state_prob_vector: np.ndarray,
    #                        emission: int) -> np.ndarray:
    #     """
    #     Compute the next state probability vector given an emission.
        
    #     This properly tracks the hidden state through the sequence.
        
    #     Args:
    #         process: Process object with transition_matrix
    #         state_prob_vector: Current state probability vector (num_states,)
    #         emission: Emitted token (emission index)
            
    #     Returns:
    #         Next state probability vector (num_states,)
    #     """
    #     try:
    #         # Import the helper function from process module
    #         from epsilon_transformers.process.Process import _compute_next_distribution
            
    #         # Compute next state given emission
    #         next_state = _compute_next_distribution(
    #             process.transition_matrix, state_prob_vector, emission
    #         )
    #         return next_state
    #     except ImportError:
    #         # Fallback: compute directly using the formula
    #         T = process.transition_matrix  # (vocab_len, num_states, num_states)
    #         X_next = np.einsum("sd, s -> d", T[emission], state_prob_vector)
    #         X_next = X_next / np.sum(X_next) if np.sum(X_next) != 0 else X_next
    #         return X_next
    def _compute_next_state(self, T, state_prob_vector, emission, xp):        
        X_next = xp.einsum("sd, s -> d", T[int(emission)], state_prob_vector)    
        x_sum = xp.sum(X_next)
        if x_sum != 0:
            X_next = X_next / x_sum  
        return X_next

    # def _get_initial_state(self, process) -> np.ndarray:
    #     """Get the initial state probability vector from process steady state."""
    #     return process.steady_state_vector
    def _get_initial_state(self, process, xp) -> object:
        vec = process.steady_state_vector
        if xp != np and not isinstance(vec, xp.ndarray):
            vec = xp.asarray(vec)
        
        # Ensure it's 1D to avoid broadcasting issues during tiling
        if vec.ndim > 1:
            vec = vec.flatten()
            
        return vec
    
    # def compute_kl_divergence_batch(self,
    #                                model_logits: torch.Tensor,
    #                                sequences: torch.Tensor,
    #                                process) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Compute KL divergence by tracking state through sequence.
        
    #     For each position in sequence:
    #     1. Start from initial state (or previous state)
    #     2. Compute emission probabilities for current state
    #     3. Update state based on observed token
    #     4. Compare with model prediction
        
    #     Args:
    #         model_logits: (batch_size, seq_len, vocab_size) model output logits
    #         sequences: (batch_size, seq_len) token sequences
    #         process: Process object with transition_matrix and steady_state_vector
            
    #     Returns:
    #         kl_per_position: (seq_len - 1,) - averaged KL at each position across batch
    #         kl_all_values: all individual KL values flattened
    #     """
    #     xp=np
    #     if 'cupy' in str(type(model_logits)):
    #         xp=cp

    #     batch_size, seq_len, vocab_size = model_logits.shape
    #     device = model_logits.device
        
    #     # Store KL values at each position
    #     current_states=np.tile(self._get_initial_state(process),(batch_size,1))#(batch_size,num_states)
    #     kl_per_position = torch.zeros(seq_len - 1, device=device)
    #     kl_all_values = []
    #     for pos in range(1,seq_len):
    #         emission_probs=np.stack([self._compute_emission_probabilities(process,current_states[b]) for b in range(batch_size)])
    #         gt_dist=torch.from_numpy(emission_probs).float().to(device)
    #         gt_dist=gt_dist/(gt_dist.sum(dim=1,keepdim=True))
    #         model_probs=F.softmax(model_logits[:,pos,:],dim=1)
    #         kl_batch=torch.sum(gt_dist*(torch.log(gt_dist+1e-10)-torch.log(model_probs+1e-10)),dim=-1)
    #         kl_per_position[pos-1]=kl_batch.mean()
    #         kl_all_values.append(kl_batch.detach().cpu())
    #         emissions=sequences[:,pos-1].cpu().numpy()#(batch,)
    #         next_states = np.stack([self._compute_next_state(process, current_states[b], int(emissions[b]))
    #         for b in range(batch_size)])
    #         current_states = next_states  
    #     kl_all_values = torch.cat(kl_all_values, dim=0)

        
    #     return kl_per_position, kl_all_values
    def compute_kl_divergence_batch(self,
                                   model_logits,
                                   sequences,
                                   process) -> Tuple[object, object]:
        """
        Compute KL divergence by tracking state through sequence.
        
        For each position in sequence:
        1. Start from initial state (or previous state)
        2. Compute emission probabilities for current state
        3. Update state based on observed token
        4. Compare with model prediction
        
        Args:
            model_logits: (batch_size, seq_len, vocab_size) model output logits
            sequences: (batch_size, seq_len) token sequences
            process: Process object with transition_matrix and steady_state_vector
            
        Returns:
            kl_per_position: (seq_len - 1,) - averaged KL at each position across batch
            kl_all_values: all individual KL values flattened
        """
        xp=np
        if 'cupy' in str(type(model_logits)):
            import cupy as cp
            xp=cp

        batch_size, seq_len, vocab_size = model_logits.shape
        device = model_logits.device
        T=process.transition_matrix
        if xp!=np and not isinstance(T, xp.ndarray):
            T=xp.asarray(T)


        # Store KL values at each position
        current_states=xp.tile(self._get_initial_state(process,xp),(batch_size,1))#(batch_size,num_states)
        kl_per_position = xp.zeros(seq_len - 1)
        kl_all_values = []
        # Pre-compute Marginal T for Emission Probabilities
        # T: [E, S, D] -> sum over D -> [E, S]
        # This represents P(emission | state)
        T_marginal = T.sum(axis=2)
        # Transpose to [S, E] for easy matrix multiplication
        T_marginal_T = T_marginal.T 

        for pos in range(1,seq_len):
            # --- A. Compute Ground Truth Distribution (Batch Vectorized) ---
            # We want P(e) for each batch = P(s) * P(e|s)
            # current_states: [B, S]
            # T_marginal_T:   [S, E]
            # Result:         [B, E] (Batch, Vocab)
            
            emission_probs = xp.matmul(current_states, T_marginal_T)
            
            # Normalize
            emission_probs = emission_probs / (emission_probs.sum(axis=1, keepdims=True) + 1e-10)
            gt_dist = emission_probs # [B, V]
            logit_batch=model_logits[:,pos,:]
            logit_max = xp.max(logit_batch, axis=1, keepdims=True)
            exp_logits = xp.exp(logit_batch - logit_max)
            model_probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)

            # --- C. Compute KL ---
            term1 = xp.log(gt_dist + 1e-10)
            term2 = xp.log(model_probs + 1e-10)
            kl_batch = xp.sum(gt_dist * (term1 - term2), axis=-1)
            
            kl_per_position[pos-1] = xp.mean(kl_batch)
            kl_all_values.append(kl_batch) 

            emissions = sequences[:, pos-1] # [B]
            
            # T[emissions] -> [B, S, D]
            T_selected = T[emissions] 
            
            # Next state: [B, D]
            next_states = xp.einsum("bsd, bs -> bd", T_selected, current_states)
            
            # Normalize
            next_states = next_states / (next_states.sum(axis=1, keepdims=True) + 1e-10)
            current_states = next_states

            
        if len(kl_all_values) > 0:
            if xp != np:
                kl_all_values = xp.concatenate(kl_all_values, axis=0)
            else:
                kl_all_values = np.concatenate(kl_all_values, axis=0)    
        
        return kl_per_position, kl_all_values


# def compute_markov_kl_divergence(model_logits: torch.Tensor,
#                                  sequences: torch.Tensor,
#                                  process,
#                                  analyzer: Optional[MarkovKLAnalyzer] = None,
#                                  return_per_position: bool = True) -> Dict[str, float]:
#     """
#     Compute Markov process KL divergence metrics for model validation.
    
#     Properly tracks the hidden state through the sequence and computes emission 
#     probabilities at each step.
    
#     Returns both:
#     - Per-position KL divergences: Average KL at each position in sequence
#     - Overall average KL divergence: Mean KL across all positions and batches
    
#     Args:
#         model_logits: Tensor of shape (batch_size, seq_len, vocab_size)
#         sequences: Tensor of shape (batch_size, seq_len) containing token IDs
#         process: Process object with transition_matrix and steady_state_vector
#         analyzer: Optional MarkovKLAnalyzer instance (creates new if None)
#         return_per_position: If True, return per-position metrics; if False, only overall
        
#     Returns:
#         Dictionary with KL divergence metrics:
#         - kl_div_markov: Overall mean KL divergence
#         - kl_div_markov_pos_{i}: KL divergence at each position (if return_per_position=True)
        
#     Example:
#         >>> model_logits = torch.randn(32, 64, 512)  # (batch, seq_len, vocab)
#         >>> sequences = torch.randint(0, 512, (32, 64))
#         >>> metrics = compute_markov_kl_divergence(model_logits, sequences, process)
#         >>> print(metrics["kl_div_markov"])  # Overall KL divergence
#     """
#     if analyzer is None:
#         analyzer = MarkovKLAnalyzer(model_logits.shape[-1], model_logits.shape[1])
    
#     kl_per_position, kl_all_values = analyzer.compute_kl_divergence_batch(
#         model_logits, sequences, process
#     )
    
#     results = {
#         "kl_div_markov": float(kl_all_values.mean().item()),
#     }
    
#     # Add per-position metrics if requested
#     if return_per_position:
#         for pos_idx, kl_at_pos in enumerate(kl_per_position):
#             position = pos_idx + 1  # Position 1 is index 0
#             results[f"kl_div_markov_pos_{position}"] = float(kl_at_pos.item())
    
#     return results

def compute_markov_kl_divergence(model_logits,
                                 sequences,
                                 process,
                                 analyzer: Optional[MarkovKLAnalyzer] = None,
                                 return_per_position: bool = True) -> Dict[str, float]:
    """
    Compute Markov process KL divergence metrics for model validation.
    """
    if analyzer is None:
        analyzer = MarkovKLAnalyzer(model_logits.shape[-1], model_logits.shape[1])
    
    kl_per_position, kl_all_values = analyzer.compute_kl_divergence_batch(
        model_logits, sequences, process
    )
    
    results = {
        "kl_div_markov": float(kl_all_values.mean()),
    }
    
    if return_per_position:
        if 'cupy' in str(type(kl_per_position)):
            kl_per_pos_cpu = kl_per_position.get()
        else:
            kl_per_pos_cpu = kl_per_position

        for pos_idx, kl_at_pos in enumerate(kl_per_pos_cpu):
            position = pos_idx + 1
            results[f"kl_div_markov_pos_{position}"] = float(kl_at_pos)
    
    return results