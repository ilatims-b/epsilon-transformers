
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class NGramAnalyzer:
    """
    Highly optimized N-Gram Analyzer using dense PyTorch tensors.
    Avoids dictionaries and Python loops for maximum GPU throughput.
    """
    
    def __init__(self, vocab_size: int, n_grams: List[int] = None):
        self.vocab_size = vocab_size
        self.n_grams = n_grams if n_grams is not None else [1, 2, 3]
        
        # Store probability tables directly on the device as tensors
        # Key: n -> Value: Tensor of shape [V, V, ..., V] (n times)
        self.prob_tables: Dict[int, torch.Tensor] = {}
        self.device = torch.device('cpu') 

    def build_from_sequences(self, sequences: torch.Tensor) -> None:
        """
        Vectorized build of N-Gram counts.
        Args:
            sequences: LongTensor of shape [Batch, Length]
        """
        self.device = sequences.device
        sequences = sequences.long()
        
        print(f"[NGramAnalyzer] Building statistics on {self.device} for N={self.n_grams}...")

        for n in self.n_grams:
            # 1. Extract all windows of size N
            # unfold -> [Batch, Num_Windows, n]
            windows = sequences.unfold(1, n, 1)
            
            # 2. Flatten -> [Total_Samples, n]
            windows_flat = windows.reshape(-1, n)
            
            # 3. Linearize indices: Index = w_0 * V^(n-1) + ... + w_n-1 * V^0
            powers = torch.tensor(
                [self.vocab_size ** i for i in range(n - 1, -1, -1)], 
                device=self.device, 
                dtype=torch.long
            )
            
            flat_indices = (windows_flat * powers).sum(dim=1)
            
            # 4. Count (Histogram)
            total_bins = self.vocab_size ** n
            counts = torch.bincount(flat_indices, minlength=total_bins).float()
            
            # 5. Reshape back to [V, V, ..., V]
            dense_counts = counts.view([self.vocab_size] * n)
            
            # 6. Normalize over last dimension (next token)
            dense_counts += 1e-10 # Smoothing
            sums = dense_counts.sum(dim=-1, keepdim=True)
            probs = dense_counts / sums
            
            self.prob_tables[n] = probs
        
        print(f"[NGramAnalyzer] Build complete.")

    def compute_kl_divergence_batch(self, model_logits: torch.Tensor, sequences: torch.Tensor, n: int):
        """
        Vectorized KL computation for a batch.
        """
        # Ensure inputs are on same device
        if sequences.device != self.device:
            sequences = sequences.to(self.device)
        if model_logits.device != self.device:
            model_logits = model_logits.to(self.device)

        batch_size, seq_len, vocab_size = model_logits.shape
        prob_table = self.prob_tables[n]

        # Valid predictions start at index n-1 (needing n-1 context)
        valid_seq_len = seq_len - (n - 1)
        if valid_seq_len <= 0:
            return torch.tensor([]), torch.tensor([])
        valid_logits=model_logits[:, n-1:, :]    

        # --- 1. Prepare Ground Truth ---
        if n == 1:
            # expand to [Batch, Len, V]
            gt_dist = prob_table.view(1, 1, -1).expand(batch_size, seq_len, vocab_size)
            # For n=1, we validly predict from index 0 to end
            valid_logits = model_logits
        else:
            # Extract context windows of size n-1 from INPUT sequences
            # These windows act as indices into the prob table
            context_windows = sequences.unfold(1, n - 1, 1)
            # context_windows shape: [Batch, Valid_Len, n-1]
            required_len=valid_logits.size(1)
            context_windows=context_windows[:, :required_len, :]

            ctx_flat = context_windows.reshape(-1, n - 1) #[Total_Preds, n-1]
            
            # Lookup: table[col0, col1, ...]
            indices = ctx_flat.unbind(dim=1)
            gt_dist_flat = prob_table[indices]
            
            gt_dist = gt_dist_flat.view(batch_size, -1, vocab_size)#[Batch, Valid_Len, V]

        # --- 2. Compute Model Probabilities ---
        model_log_probs = F.log_softmax(valid_logits, dim=-1)
        
        # --- 3. KL Computation ---
        # KL = sum( P * (log P - log Q) )
        gt_dist = gt_dist + 1e-12
        gt_log_probs = torch.log(gt_dist)
        
        kl_elementwise = gt_dist * (gt_log_probs - model_log_probs)
        kl_per_token = kl_elementwise.sum(dim=-1) # [Batch, Valid_Len]
        
        kl_per_position = kl_per_token.mean(dim=0)
        kl_all_values = kl_per_token.view(-1)
        
        return kl_per_position, kl_all_values


def compute_ngram_kl_divergence(model_logits, sequences, ngram_analyzer, n_values=None, return_per_position=True):
    if n_values is None:
        n_values = ngram_analyzer.n_grams
    
    results = {}
    for n in n_values:
        kl_per_position, kl_all_values = ngram_analyzer.compute_kl_divergence_batch(
            model_logits, sequences, n
        )
        
        if kl_all_values.numel() > 0:
            results[f"kl_div_ngram_{n}"] = float(kl_all_values.mean().item())
        else:
            results[f"kl_div_ngram_{n}"] = 0.0
        
        if return_per_position and kl_per_position.numel() > 0:
            kl_pos_list = kl_per_position.tolist()
            for i, val in enumerate(kl_pos_list):
                # Offset: if n=2, first metric is for pos 1 (0-indexed)
                real_pos = (n - 1) + i
                results[f"kl_div_ngram_{n}_pos_{real_pos}"] = val
    
    return results


# import torch
# import torch.nn.functional as F
# from collections import defaultdict
# from typing import Dict, Tuple, List, Optional
# import numpy as np

# class NGramAnalyzer:
#     """Analyzes n-gram probabilities and computes KL divergence metrics."""
    
#     def __init__(self, vocab_size: int, n_grams: List[int] = None, seq_len: int = None):
#         self.vocab_size = vocab_size
#         self.n_grams = n_grams if n_grams is not None else [1, 2, 3]
#         self.seq_len = seq_len
#         self.prob_tables: Dict[int, torch.Tensor]={}
#         self.device=torch.device('cpu')
#         # self.ngram_counts = {n: defaultdict(lambda: defaultdict(int)) for n in self.n_grams}
#         # self.unigram_counts = defaultdict(int)
#         self.total_sequences = 0
        
#         # Cache for dense probability tables (lazy loaded)
#         self.dense_tables = {} 

#     def build_from_sequences(self, sequences) -> None:
#         """Build n-gram model from sequences."""
#         # Keep build logic on CPU/NumPy as it uses dicts
#         if hasattr(sequences, 'cpu'):
#             sequences = sequences.cpu().numpy()
#         elif 'cupy' in str(type(sequences)):
#             sequences = sequences.get()
            
#         for seq in sequences:
#             for pos in range(1, len(seq)):
#                 next_token = int(seq[pos])
#                 self.unigram_counts[next_token] += 1
#                 for n in self.n_grams:
#                     if pos >= n:
#                         context = tuple(seq[max(0, pos - n + 1):pos].astype(int))
#                         self.ngram_counts[n][context][next_token] += 1
        
#         self.total_sequences = len(sequences)
#         self.dense_tables = {} # Reset cache after rebuild

#     def _get_dense_table(self, n: int, xp):
#         """
#         Convert sparse N-gram counts to a dense probability tensor on the backend (xp).
#         """
#         # Check cache first
#         cache_key = (n, 'cupy' if xp != np else 'numpy')
#         if cache_key in self.dense_tables:
#             return self.dense_tables[cache_key]

#         print(f"Building dense table for N={n} on {xp.__name__}...")
        
#         # Shape: [Vocab, Vocab, ... (n times)]
#         # Last dim is 'next_token', previous dims are context
#         shape = [self.vocab_size] * n
#         dense_counts = xp.zeros(shape, dtype=xp.float32) + 1e-10 # Add smoothing
        
#         # Populate dense table (Iterate CPU dict)
#         counts_dict = self.ngram_counts[n]
        
#         # Optimized: Build on CPU NumPy first, then move to xp
#         if xp != np:
#             cpu_counts = np.zeros(shape, dtype=np.float32) + 1e-10
#             for context, token_counts in counts_dict.items():
#                 for token, count in token_counts.items():
#                     full_index = context + (token,)
#                     cpu_counts[full_index] += count # += to add to smoothing
            
#             dense_counts = xp.asarray(cpu_counts)
#         else:
#             # Already on CPU
#              for context, token_counts in counts_dict.items():
#                 for token, count in token_counts.items():
#                     full_index = context + (token,)
#                     dense_counts[full_index] += count

#         # Normalize last dimension to get probabilities P(next | context)
#         # Sum over last axis (next_token)
#         sums = dense_counts.sum(axis=-1, keepdims=True)
#         dense_probs = dense_counts / sums
        
#         self.dense_tables[cache_key] = dense_probs
#         return dense_probs

#     def compute_kl_divergence_batch(self, model_logits, sequences, n: int):
#         xp = np
#         if 'cupy' in str(type(model_logits)):
#             import cupy as cp
#             xp = cp

#         batch_size, seq_len, vocab_size = model_logits.shape
        
#         # 1. Get Dense Probability Table [V, V, ... V]
#         prob_table = self._get_dense_table(n, xp)
        
#         kl_per_position_list = [] # Using list since size varies based on N
#         kl_all_values = []
        
#         for pos in range(1, seq_len):
#             # --- A. Get Ground Truth (Vectorized Lookup) ---
#             if pos < n:
#                 # Not enough context for N-gram, skip completely
#                 continue
                
#             # Get Context Indices
#             if n == 1:
#                 # Unigram: No context. Table is [V].
#                 gt_dist = xp.tile(prob_table, (batch_size, 1))
#             else:
#                 # N > 1. Context size is n-1.
#                 indices = []
#                 for i in range(n-1, 0, -1): # n-1 down to 1
#                     idx = sequences[:, pos-i] # [Batch]
#                     indices.append(idx)
                    
#                 # Advanced Indexing: prob_table[idx_1, idx_2, ..., :] -> Shape [Batch, Vocab]
#                 gt_dist = prob_table[tuple(indices)] 
            
#             # --- B. Model Probs ---
#             logit_batch = model_logits[:, pos, :]
#             logit_max = xp.max(logit_batch, axis=1, keepdims=True)
#             exp_logits = xp.exp(logit_batch - logit_max)
#             model_probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)
            
#             # --- C. KL ---
#             term1 = xp.log(gt_dist + 1e-10)
#             term2 = xp.log(model_probs + 1e-10)
#             kl_batch = xp.sum(gt_dist * (term1 - term2), axis=-1)
            
#             # Append mean for this position
#             kl_per_position_list.append(xp.mean(kl_batch))
            
#             # Append all batch values
#             kl_all_values.append(kl_batch)

#         # Convert per-position list to array
#         if len(kl_per_position_list) > 0:
#             kl_per_position = xp.stack(kl_per_position_list)
#         else:
#             kl_per_position = xp.array([])

#         # Concatenate all values
#         if len(kl_all_values) > 0:
#             if xp != np:
#                 kl_all_values = xp.concatenate(kl_all_values, axis=0)
#             else:
#                 kl_all_values = np.concatenate(kl_all_values, axis=0)
#         else:
#              kl_all_values = xp.array([])
                
#         return kl_per_position, kl_all_values

# def compute_ngram_kl_divergence(model_logits, sequences, ngram_analyzer, n_values=None, return_per_position=True):
#     if n_values is None:
#         n_values = ngram_analyzer.n_grams
    
#     results = {}
#     for n in n_values:
#         kl_per_position, kl_all_values = ngram_analyzer.compute_kl_divergence_batch(
#             model_logits, sequences, n
#         )
        
#         # Overall mean KL
#         if kl_all_values.size > 0:
#             results[f"kl_div_ngram_{n}"] = float(kl_all_values.mean())
#         else:
#             results[f"kl_div_ngram_{n}"] = 0.0
        
#         # Per-position metrics
#         if return_per_position:
#             if 'cupy' in str(type(kl_per_position)):
#                 kl_per_pos_cpu = kl_per_position.get()
#             else:
#                 kl_per_pos_cpu = kl_per_position

#             for i, kl_at_pos in enumerate(kl_per_pos_cpu):
#                 # Since we skipped positions < n, the first index corresponds to pos = n
#                 # Example: n=2. Loop starts at pos=2.
#                 # i=0 -> pos=2
#                 # i=1 -> pos=3
#                 real_position = n + i
#                 results[f"kl_div_ngram_{n}_pos_{real_position}"] = float(kl_at_pos)
    
#     return results