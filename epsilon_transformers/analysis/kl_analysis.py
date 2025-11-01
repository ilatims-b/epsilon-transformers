"""
Corrected KL divergence analysis - properly track state through sequence.

This module computes KL divergence against ground truth Markov process distributions
by correctly tracking the hidden state through the sequence using _compute_next_distribution.
"""

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
    
    def _compute_emission_probabilities(self,
                                       process,
                                       state_prob_vector: np.ndarray) -> np.ndarray:
        """
        Compute emission probabilities for the current state.
        
        This uses the internal _compute_emission_probabilities function from Process.
        
        Args:
            process: Process object with transition_matrix
            state_prob_vector: Current state probability vector (num_states,)
            
        Returns:
            Emission probability tensor of shape (vocab_size,)
        """
        try:
            # Import the helper function from process module
            from epsilon_transformers.process.Process import _compute_emission_probabilities
            
            # Compute emission probs given current state
            emission_probs = _compute_emission_probabilities(process, state_prob_vector)
            return emission_probs
        except ImportError:
            # Fallback: compute directly using the formula
            T = process.transition_matrix  # (vocab_len, num_states, num_states)
            emission_probs = np.einsum("s,esd->ed", state_prob_vector, T).sum(axis=1)
            emission_probs /= emission_probs.sum()
            return emission_probs
    
    def _compute_next_state(self,
                           process,
                           state_prob_vector: np.ndarray,
                           emission: int) -> np.ndarray:
        """
        Compute the next state probability vector given an emission.
        
        This properly tracks the hidden state through the sequence.
        
        Args:
            process: Process object with transition_matrix
            state_prob_vector: Current state probability vector (num_states,)
            emission: Emitted token (emission index)
            
        Returns:
            Next state probability vector (num_states,)
        """
        try:
            # Import the helper function from process module
            from epsilon_transformers.process.Process import _compute_next_distribution
            
            # Compute next state given emission
            next_state = _compute_next_distribution(
                process.transition_matrix, state_prob_vector, emission
            )
            return next_state
        except ImportError:
            # Fallback: compute directly using the formula
            T = process.transition_matrix  # (vocab_len, num_states, num_states)
            X_next = np.einsum("sd, s -> d", T[emission], state_prob_vector)
            X_next = X_next / np.sum(X_next) if np.sum(X_next) != 0 else X_next
            return X_next
    
    def _get_initial_state(self, process) -> np.ndarray:
        """Get the initial state probability vector from process steady state."""
        return process.steady_state_vector
    
    def compute_kl_divergence_batch(self,
                                   model_logits: torch.Tensor,
                                   sequences: torch.Tensor,
                                   process) -> Tuple[torch.Tensor, torch.Tensor]:
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
        batch_size, seq_len, vocab_size = model_logits.shape
        device = model_logits.device
        
        # Store KL values at each position
        kl_at_position = {}
        kl_all_values = []
        
        # Process each sample in batch
        for batch_idx in range(batch_size):
            seq = sequences[batch_idx].cpu().numpy()
            
            # Initialize with steady state
            current_state = self._get_initial_state(process)
            
            # Process each position in sequence
            for pos in range(1, seq_len):
                # Get ground truth emission probabilities for current state
                emission_probs = self._compute_emission_probabilities(process, current_state)
                
                # Convert to tensor
                gt_dist = torch.from_numpy(emission_probs).float()
                gt_dist = gt_dist / (gt_dist.sum() + 1e-10)  # Ensure normalized
                gt_dist = gt_dist.to(device)
                
                # Get model probabilities for next token
                model_logit = model_logits[batch_idx, pos, :]
                model_probs = F.softmax(model_logit, dim=0)
                
                # Compute KL divergence: KL(ground_truth || model)
                # KL(P||Q) = sum(P * (log(P) - log(Q)))
                kl = torch.sum(
                    gt_dist * (torch.log(gt_dist + 1e-10) - torch.log(model_probs + 1e-10))
                )
                
                kl_value = kl.item()
                
                # Store at this position
                if pos not in kl_at_position:
                    kl_at_position[pos] = []
                
                kl_at_position[pos].append(kl_value)
                kl_all_values.append(kl_value)
                
                # Update state based on observed token at position pos-1
                # (we're predicting token at pos, so we use tokens up to pos-1)
                observed_token = int(seq[pos - 1])
                current_state = self._compute_next_state(process, current_state, observed_token)
        
        # Average KL at each position across batch
        kl_per_position = torch.zeros(seq_len - 1)
        for pos in range(1, seq_len):
            if pos in kl_at_position and len(kl_at_position[pos]) > 0:
                kl_per_position[pos - 1] = np.mean(kl_at_position[pos])
        
        return kl_per_position, torch.tensor(kl_all_values, device='cpu')


def compute_markov_kl_divergence(model_logits: torch.Tensor,
                                 sequences: torch.Tensor,
                                 process,
                                 analyzer: Optional[MarkovKLAnalyzer] = None,
                                 return_per_position: bool = True) -> Dict[str, float]:
    """
    Compute Markov process KL divergence metrics for model validation.
    
    Properly tracks the hidden state through the sequence and computes emission 
    probabilities at each step.
    
    Returns both:
    - Per-position KL divergences: Average KL at each position in sequence
    - Overall average KL divergence: Mean KL across all positions and batches
    
    Args:
        model_logits: Tensor of shape (batch_size, seq_len, vocab_size)
        sequences: Tensor of shape (batch_size, seq_len) containing token IDs
        process: Process object with transition_matrix and steady_state_vector
        analyzer: Optional MarkovKLAnalyzer instance (creates new if None)
        return_per_position: If True, return per-position metrics; if False, only overall
        
    Returns:
        Dictionary with KL divergence metrics:
        - kl_div_markov: Overall mean KL divergence
        - kl_div_markov_pos_{i}: KL divergence at each position (if return_per_position=True)
        
    Example:
        >>> model_logits = torch.randn(32, 64, 512)  # (batch, seq_len, vocab)
        >>> sequences = torch.randint(0, 512, (32, 64))
        >>> metrics = compute_markov_kl_divergence(model_logits, sequences, process)
        >>> print(metrics["kl_div_markov"])  # Overall KL divergence
    """
    if analyzer is None:
        analyzer = MarkovKLAnalyzer(model_logits.shape[-1], model_logits.shape[1])
    
    kl_per_position, kl_all_values = analyzer.compute_kl_divergence_batch(
        model_logits, sequences, process
    )
    
    results = {
        "kl_div_markov": float(kl_all_values.mean().item()),
    }
    
    # Add per-position metrics if requested
    if return_per_position:
        for pos_idx, kl_at_pos in enumerate(kl_per_position):
            position = pos_idx + 1  # Position 1 is index 0
            results[f"kl_div_markov_pos_{position}"] = float(kl_at_pos.item())
    
    return results