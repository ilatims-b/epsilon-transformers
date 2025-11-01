"""
N-gram frequency analysis and KL divergence computation for validation sets.

This module computes n-gram probabilities from a dataset and calculates the KL divergence
between model predictions and n-gram baselines. Similar to Kaggle metamsp analysis but adapted
for epsilon-transformers where data generation is done on-the-fly.

Returns both:
- Per-position KL divergences: Average KL at each position in sequence
- Overall average KL divergence: Mean KL across all positions
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Tuple, List
import numpy as np


class NGramAnalyzer:
    """Analyzes n-gram probabilities and computes KL divergence metrics."""
    
    def __init__(self, vocab_size: int, n_grams: List[int] = None, seq_len: int = None):
        """
        Initialize the n-gram analyzer.
        
        Args:
            vocab_size: Size of the vocabulary
            n_grams: List of n values to compute (e.g., [1, 2, 3])
            seq_len: Optional expected sequence length (for pre-allocation)
        """
        self.vocab_size = vocab_size
        self.n_grams = n_grams if n_grams is not None else [1, 2, 3]
        self.seq_len = seq_len
        
        # Store counts for each n-gram level
        # Structure: {n: {context_tuple: {token: count}}}
        self.ngram_counts = {n: defaultdict(lambda: defaultdict(int)) for n in self.n_grams}
        
        # Store unigram counts separately for normalization
        self.unigram_counts = defaultdict(int)
        
        self.total_sequences = 0
    
    def build_from_sequences(self, sequences: torch.Tensor) -> None:
        """
        Build n-gram model from sequences.
        
        Args:
            sequences: Tensor of shape (batch_size, seq_len) containing token IDs
        """
        sequences = sequences.cpu().numpy()
        
        for seq in sequences:
            for pos in range(1, len(seq)):
                next_token = int(seq[pos])
                self.unigram_counts[next_token] += 1
                
                # Build n-grams of various orders
                for n in self.n_grams:
                    if pos >= n:
                        # Context is the previous (n-1) tokens
                        context = tuple(seq[max(0, pos - n + 1):pos].astype(int))
                        self.ngram_counts[n][context][next_token] += 1
        
        self.total_sequences = len(sequences)
    
    def get_next_token_probabilities(self, context: Tuple, n: int) -> Dict[int, float]:
        """
        Get probability distribution over next tokens given a context for n-gram of size n.
        
        Args:
            context: Tuple of previous token IDs
            n: N-gram size
            
        Returns:
            Dictionary mapping token IDs to probabilities
        """
        if n not in self.n_grams:
            raise ValueError(f"n={n} not in configured n_grams: {self.n_grams}")
        
        # For n-grams, we need context of size (n-1)
        context_key = context[-(n-1):] if len(context) >= (n-1) else context
        
        counts = self.ngram_counts[n].get(context_key, defaultdict(int))
        total = sum(counts.values())
        
        if total == 0:
            # Fallback to uniform distribution if context not seen
            return {i: 1.0 / self.vocab_size for i in range(self.vocab_size)}
        
        probs = {token: count / total for token, count in counts.items()}
        
        # Add small probability for unseen tokens
        seen_tokens = set(probs.keys())
        unseen_prob = 1e-10
        for i in range(self.vocab_size):
            if i not in seen_tokens:
                probs[i] = unseen_prob
        
        return probs
    
    def compute_kl_divergence_batch(self, 
                                    model_logits: torch.Tensor,
                                    sequences: torch.Tensor,
                                    n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence between model predictions and n-gram probabilities.
        
        Args:
            model_logits: Tensor of shape (batch_size, seq_len, vocab_size) with model logits
            sequences: Tensor of shape (batch_size, seq_len) with token IDs
            n: N-gram size to use
            
        Returns:
            Tuple of:
            - kl_per_position: Tensor of shape (seq_len - 1,) with KL at each position
            - kl_all_values: Tensor of all KL divergence values across batch and positions
        """
        batch_size, seq_len, vocab_size = model_logits.shape
        sequences_np = sequences.cpu().numpy()
        
        kl_values_per_position = {pos: [] for pos in range(1, seq_len)}
        kl_all_values = []
        
        for batch_idx in range(batch_size):
            seq = sequences_np[batch_idx]
            
            for pos in range(1, seq_len):
                # Get context
                context_start = max(0, pos - n + 1)
                context = tuple(seq[context_start:pos].astype(int))
                
                # Get n-gram probabilities
                ngram_probs = self.get_next_token_probabilities(context, n)
                
                # Convert to tensor
                ngram_prob_vec = torch.zeros(vocab_size, device=model_logits.device)
                for token_id, prob in ngram_probs.items():
                    ngram_prob_vec[token_id] = prob
                
                # Normalize to ensure probabilities sum to 1
                ngram_prob_vec = ngram_prob_vec / (ngram_prob_vec.sum() + 1e-10)
                
                # Get model probabilities
                model_logit = model_logits[batch_idx, pos, :]
                model_probs = F.softmax(model_logit, dim=0)
                
                # Compute KL(ngram || model)
                # KL(P||Q) = sum(P * (log(P) - log(Q)))
                kl = torch.sum(
                    ngram_prob_vec * (torch.log(ngram_prob_vec + 1e-10) - torch.log(model_probs + 1e-10))
                )
                
                kl_value = kl.item()
                kl_values_per_position[pos].append(kl_value)
                kl_all_values.append(kl_value)
        
        # Average KL at each position across batch
        kl_per_position = torch.zeros(seq_len - 1)
        for pos in range(1, seq_len):
            if len(kl_values_per_position[pos]) > 0:
                kl_per_position[pos - 1] = np.mean(kl_values_per_position[pos])
        
        return kl_per_position, torch.tensor(kl_all_values, device='cpu')


def compute_ngram_kl_divergence(model_logits: torch.Tensor,
                                 sequences: torch.Tensor,
                                 ngram_analyzer: NGramAnalyzer,
                                 n_values: List[int] = None,
                                 return_per_position: bool = True) -> Dict[str, float]:
    """
    Compute n-gram KL divergence metrics for model validation.
    
    Returns both:
    - Per-position KL divergences: Average KL at each position in sequence
    - Overall average KL divergence: Mean KL across all positions
    
    Args:
        model_logits: Tensor of shape (batch_size, seq_len, vocab_size)
        sequences: Tensor of shape (batch_size, seq_len)
        ngram_analyzer: Initialized NGramAnalyzer with trained frequencies
        n_values: Specific n values to compute (if None, uses analyzer's n_grams)
        return_per_position: If True, return per-position metrics; if False, only overall
        
    Returns:
        Dictionary with KL divergence metrics keyed by n-gram size, including:
        - kl_div_ngram_{n}: Overall mean KL for n-gram of size n
        - kl_div_ngram_{n}_pos_{i}: KL divergence at each position (if return_per_position=True)
    """
    if n_values is None:
        n_values = ngram_analyzer.n_grams
    
    results = {}
    for n in n_values:
        kl_per_position, kl_all_values = ngram_analyzer.compute_kl_divergence_batch(
            model_logits, sequences, n
        )
        
        # Overall mean KL for this n-gram
        results[f"kl_div_ngram_{n}"] = float(kl_all_values.mean().item())
        
        # Per-position metrics if requested
        if return_per_position:
            for pos_idx, kl_at_pos in enumerate(kl_per_position):
                position = pos_idx + 1  # Position 1 is index 0
                results[f"kl_div_ngram_{n}_pos_{position}"] = float(kl_at_pos.item())
    
    return results