

# from re import X
# import torch
# import torch.nn.functional as F
# from collections import defaultdict
# from typing import Dict, Tuple, List
# import numpy as np


# class NGramAnalyzer:
#     """Analyzes n-gram probabilities and computes KL divergence metrics."""
    
#     def __init__(self, vocab_size: int, n_grams: List[int] = None, seq_len: int = None):
#         """
#         Initialize the n-gram analyzer.
        
#         Args:
#             vocab_size: Size of the vocabulary
#             n_grams: List of n values to compute (e.g., [1, 2, 3])
#             seq_len: Optional expected sequence length (for pre-allocation)
#         """
#         self.vocab_size = vocab_size
#         self.n_grams = n_grams if n_grams is not None else [1, 2, 3]
#         self.seq_len = seq_len
        
#         # Store counts for each n-gram level
#         # Structure: {n: {context_tuple: {token: count}}}
#         self.ngram_counts = {n: defaultdict(lambda: defaultdict(int)) for n in self.n_grams}
        
#         # Store unigram counts separately for normalization
#         self.unigram_counts = defaultdict(int)
        
#         self.total_sequences = 0
    
#     def build_from_sequences(self, sequences: torch.Tensor) -> None:
#         """
#         Build n-gram model from sequences.
        
#         Args:
#             sequences: Tensor of shape (batch_size, seq_len) containing token IDs
#         """
#         if hasattr(sequences,'cpu'):
#             sequences = sequences.cpu().numpy()
#         elif 'cupy' in str(type(sequences)):
#             sequences=sequences.get()   
        
#         for seq in sequences:
#             for pos in range(1, len(seq)):
#                 next_token = int(seq[pos])
#                 self.unigram_counts[next_token] += 1
                
#                 # Build n-grams of various orders
#                 for n in self.n_grams:
#                     if pos >= n:
#                         # Context is the previous (n-1) tokens
#                         context = tuple(seq[max(0, pos - n + 1):pos].astype(int))
#                         self.ngram_counts[n][context][next_token] += 1
        
#         self.total_sequences = len(sequences)
    
#     def get_next_token_probabilities(self, context: Tuple, n: int) -> Dict[int, float]:
#         """
#         Get probability distribution over next tokens given a context for n-gram of size n.
        
#         Args:
#             context: Tuple of previous token IDs
#             n: N-gram size
            
#         Returns:
#             Dictionary mapping token IDs to probabilities
#         """
#         if n not in self.n_grams:
#             raise ValueError(f"n={n} not in configured n_grams: {self.n_grams}")
        
#         # For n-grams, we need context of size (n-1)
#         context_key = context[-(n-1):] if len(context) >= (n-1) else context
        
#         counts = self.ngram_counts[n].get(context_key, defaultdict(int))
#         total = sum(counts.values())
        
#         if total == 0:
#             # Fallback to uniform distribution if context not seen
#             return {i: 1.0 / self.vocab_size for i in range(self.vocab_size)}
        
#         probs = {token: count / total for token, count in counts.items()}
        
#         # Add small probability for unseen tokens
#         seen_tokens = set(probs.keys())
#         unseen_prob = 1e-10
#         for i in range(self.vocab_size):
#             if i not in seen_tokens:
#                 probs[i] = unseen_prob
        
#         return probs
    
#     def compute_kl_divergence_batch(self, 
#                                     model_logits,
#                                     sequences ,
#                                     n: int) -> Tuple[object, object]:
#         """
#         Compute KL divergence between model predictions and n-gram probabilities.
        
#         Args:
#             model_logits: Tensor of shape (batch_size, seq_len, vocab_size) with model logits
#             sequences: Tensor of shape (batch_size, seq_len) with token IDs
#             n: N-gram size to use
            
#         Returns:
#             Tuple of:
#             - kl_per_position: Tensor of shape (seq_len - 1,) with KL at each position
#             - kl_all_values: Tensor of all KL divergence values across batch and positions
#         """
#         xp=np
#         if 'cupy' in str(type(model_logits)):
#             import cupy as cp
#             xp=cp

#         batch_size, seq_len, vocab_size = model_logits.shape
#         if xp!=np:
#             sequences_np = sequences.get()
#         else: 
#             sequences_np = sequences   
#         # sequences_np = sequences.cpu().numpy()
        
#         kl_values_per_position = {pos: [] for pos in range(1, seq_len)}
#         kl_all_values = []
        
#         for batch_idx in range(batch_size):
#             seq = sequences_np[batch_idx]
            
#             for pos in range(1, seq_len):
#                 # Get context
#                 context_start = max(0, pos - n + 1)
#                 context = tuple(seq[context_start:pos].astype(int))
                
#                 # Get n-gram probabilities
#                 ngram_probs = self.get_next_token_probabilities(context, n)
                
#                 # # Convert to tensor
#                 # ngram_prob_vec = torch.zeros(vocab_size, device=model_logits.device)
#                 ngram_prob_vec_cpu=np.zeros(vocab_size, dtype=np.float32)
#                 for token_id, prob in ngram_probs.items():
#                     ngram_prob_vec_cpu[token_id] = prob

#                 if xp!=np:
#                     ngram_prob_vec=xp.asarray(ngram_prob_vec_cpu)
#                 else:
#                     ngram_prob_vec=ngram_prob_vec_cpu        
                
#                 # Normalize to ensure probabilities sum to 1
#                 ngram_prob_vec = ngram_prob_vec / (ngram_prob_vec.sum() + 1e-10)
                
#                 # Get model probabilities
#                 model_logit = model_logits[batch_idx, pos, :]
#                 # model_probs = F.softmax(model_logit, dim=0)
#                 # Softmax
#                 logit_max = xp.max(model_logit)
#                 exp_logits = xp.exp(model_logit - logit_max)
#                 model_probs = exp_logits / xp.sum(exp_logits)
#                 # Compute KL(ngram || model)
#                 # # KL(P||Q) = sum(P * (log(P) - log(Q)))
#                 # kl = torch.sum(
#                 #     ngram_prob_vec * (torch.log(ngram_prob_vec + 1e-10) - torch.log(model_probs + 1e-10))
#                 # )
#                 # KL = sum(P * (log P - log Q))
#                 term1 = xp.log(ngram_prob_vec + 1e-10)
#                 term2 = xp.log(model_probs + 1e-10)
#                 kl = xp.sum(ngram_prob_vec * (term1 - term2))

#                 kl_value = kl.item()
#                 kl_values_per_position[pos].append(kl_value)
#                 kl_all_values.append(kl_value)
        
#         # Average KL at each position across batch
#         kl_per_position = xp.zeros(seq_len - 1)
#         for pos in range(1, seq_len):
#             if len(kl_values_per_position[pos]) > 0:
#                 kl_per_position[pos - 1] = xp.mean(xp.array(kl_values_per_position[pos]))
        
#         return kl_per_position, xp.array(kl_all_values)


# def compute_ngram_kl_divergence(model_logits,
#                                  sequences,
#                                  ngram_analyzer: NGramAnalyzer,
#                                  n_values: List[int] = None,
#                                  return_per_position: bool = True) -> Dict[str, float]:
#     """
#     Compute n-gram KL divergence metrics for model validation.
    
#     Returns both:
#     - Per-position KL divergences: Average KL at each position in sequence
#     - Overall average KL divergence: Mean KL across all positions
    
#     Args:
#         model_logits: Tensor of shape (batch_size, seq_len, vocab_size)
#         sequences: Tensor of shape (batch_size, seq_len)
#         ngram_analyzer: Initialized NGramAnalyzer with trained frequencies
#         n_values: Specific n values to compute (if None, uses analyzer's n_grams)
#         return_per_position: If True, return per-position metrics; if False, only overall
        
#     Returns:
#         Dictionary with KL divergence metrics keyed by n-gram size, including:
#         - kl_div_ngram_{n}: Overall mean KL for n-gram of size n
#         - kl_div_ngram_{n}_pos_{i}: KL divergence at each position (if return_per_position=True)
#     """
#     if n_values is None:
#         n_values = ngram_analyzer.n_grams
    
#     results = {}
#     for n in n_values:
#         kl_per_position, kl_all_values = ngram_analyzer.compute_kl_divergence_batch(
#             model_logits, sequences, n
#         )
        
#         # Overall mean KL for this n-gram
#         results[f"kl_div_ngram_{n}"] = float(kl_all_values.mean().item())
        
#         # Per-position metrics if requested
#         if return_per_position:
#             if 'cupy' in str(type(kl_per_position)):
#                 kl_per_pos_cpu=kl_per_position.get()
#             else:
#                 kl_per_pos_cpu=kl_per_position    
#             for pos_idx, kl_at_pos in enumerate(kl_per_pos_cpu):
#                 position = pos_idx + 1  # Position 1 is index 0
#                 results[f"kl_div_ngram_{n}_pos_{position}"] = float(kl_at_pos)
    
#     return results
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import numpy as np

class NGramAnalyzer:
    """Analyzes n-gram probabilities and computes KL divergence metrics."""
    
    def __init__(self, vocab_size: int, n_grams: List[int] = None, seq_len: int = None):
        self.vocab_size = vocab_size
        self.n_grams = n_grams if n_grams is not None else [1, 2, 3]
        self.seq_len = seq_len
        self.ngram_counts = {n: defaultdict(lambda: defaultdict(int)) for n in self.n_grams}
        self.unigram_counts = defaultdict(int)
        self.total_sequences = 0
        
        # Cache for dense probability tables (lazy loaded)
        self.dense_tables = {} 

    def build_from_sequences(self, sequences) -> None:
        """Build n-gram model from sequences."""
        # Keep build logic on CPU/NumPy as it uses dicts
        if hasattr(sequences, 'cpu'):
            sequences = sequences.cpu().numpy()
        elif 'cupy' in str(type(sequences)):
            sequences = sequences.get()
            
        for seq in sequences:
            for pos in range(1, len(seq)):
                next_token = int(seq[pos])
                self.unigram_counts[next_token] += 1
                for n in self.n_grams:
                    if pos >= n:
                        context = tuple(seq[max(0, pos - n + 1):pos].astype(int))
                        self.ngram_counts[n][context][next_token] += 1
        
        self.total_sequences = len(sequences)
        self.dense_tables = {} # Reset cache after rebuild

    def _get_dense_table(self, n: int, xp):
        """
        Convert sparse N-gram counts to a dense probability tensor on the backend (xp).
        """
        # Check cache first
        cache_key = (n, 'cupy' if xp != np else 'numpy')
        if cache_key in self.dense_tables:
            return self.dense_tables[cache_key]

        print(f"Building dense table for N={n} on {xp.__name__}...")
        
        # Shape: [Vocab, Vocab, ... (n times)]
        # Last dim is 'next_token', previous dims are context
        shape = [self.vocab_size] * n
        dense_counts = xp.zeros(shape, dtype=xp.float32) + 1e-10 # Add smoothing
        
        # Populate dense table (Iterate CPU dict)
        counts_dict = self.ngram_counts[n]
        
        # Optimized: Build on CPU NumPy first, then move to xp
        if xp != np:
            cpu_counts = np.zeros(shape, dtype=np.float32) + 1e-10
            for context, token_counts in counts_dict.items():
                for token, count in token_counts.items():
                    full_index = context + (token,)
                    cpu_counts[full_index] += count # += to add to smoothing
            
            dense_counts = xp.asarray(cpu_counts)
        else:
            # Already on CPU
             for context, token_counts in counts_dict.items():
                for token, count in token_counts.items():
                    full_index = context + (token,)
                    dense_counts[full_index] += count

        # Normalize last dimension to get probabilities P(next | context)
        # Sum over last axis (next_token)
        sums = dense_counts.sum(axis=-1, keepdims=True)
        dense_probs = dense_counts / sums
        
        self.dense_tables[cache_key] = dense_probs
        return dense_probs

    def compute_kl_divergence_batch(self, model_logits, sequences, n: int):
        xp = np
        if 'cupy' in str(type(model_logits)):
            import cupy as cp
            xp = cp

        batch_size, seq_len, vocab_size = model_logits.shape
        
        # 1. Get Dense Probability Table [V, V, ... V]
        prob_table = self._get_dense_table(n, xp)
        
        kl_per_position_list = [] # Using list since size varies based on N
        kl_all_values = []
        
        for pos in range(1, seq_len):
            # --- A. Get Ground Truth (Vectorized Lookup) ---
            if pos < n:
                # Not enough context for N-gram, skip completely
                continue
                
            # Get Context Indices
            if n == 1:
                # Unigram: No context. Table is [V].
                gt_dist = xp.tile(prob_table, (batch_size, 1))
            else:
                # N > 1. Context size is n-1.
                indices = []
                for i in range(n-1, 0, -1): # n-1 down to 1
                    idx = sequences[:, pos-i] # [Batch]
                    indices.append(idx)
                    
                # Advanced Indexing: prob_table[idx_1, idx_2, ..., :] -> Shape [Batch, Vocab]
                gt_dist = prob_table[tuple(indices)] 
            
            # --- B. Model Probs ---
            logit_batch = model_logits[:, pos, :]
            logit_max = xp.max(logit_batch, axis=1, keepdims=True)
            exp_logits = xp.exp(logit_batch - logit_max)
            model_probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)
            
            # --- C. KL ---
            term1 = xp.log(gt_dist + 1e-10)
            term2 = xp.log(model_probs + 1e-10)
            kl_batch = xp.sum(gt_dist * (term1 - term2), axis=-1)
            
            # Append mean for this position
            kl_per_position_list.append(xp.mean(kl_batch))
            
            # Append all batch values
            kl_all_values.append(kl_batch)

        # Convert per-position list to array
        if len(kl_per_position_list) > 0:
            kl_per_position = xp.stack(kl_per_position_list)
        else:
            kl_per_position = xp.array([])

        # Concatenate all values
        if len(kl_all_values) > 0:
            if xp != np:
                kl_all_values = xp.concatenate(kl_all_values, axis=0)
            else:
                kl_all_values = np.concatenate(kl_all_values, axis=0)
        else:
             kl_all_values = xp.array([])
                
        return kl_per_position, kl_all_values

def compute_ngram_kl_divergence(model_logits, sequences, ngram_analyzer, n_values=None, return_per_position=True):
    if n_values is None:
        n_values = ngram_analyzer.n_grams
    
    results = {}
    for n in n_values:
        kl_per_position, kl_all_values = ngram_analyzer.compute_kl_divergence_batch(
            model_logits, sequences, n
        )
        
        # Overall mean KL
        if kl_all_values.size > 0:
            results[f"kl_div_ngram_{n}"] = float(kl_all_values.mean())
        else:
            results[f"kl_div_ngram_{n}"] = 0.0
        
        # Per-position metrics
        if return_per_position:
            if 'cupy' in str(type(kl_per_position)):
                kl_per_pos_cpu = kl_per_position.get()
            else:
                kl_per_pos_cpu = kl_per_position

            for i, kl_at_pos in enumerate(kl_per_pos_cpu):
                # Since we skipped positions < n, the first index corresponds to pos = n
                # Example: n=2. Loop starts at pos=2.
                # i=0 -> pos=2
                # i=1 -> pos=3
                real_position = n + i
                results[f"kl_div_ngram_{n}_pos_{real_position}"] = float(kl_at_pos)
    
    return results
