
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
        self.count_tables:Dict[int,torch.Tensor]={}
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
            self.count_tables[n]=dense_counts
        self.build_prob_tables_from_counts()
        print(f"[Ngramanalyzer]Build complete.")
    def build_prob_tables_from_counts(self):
        self.prob_tables={}
        for n in self.count_tables:
            if n not in self.count_tables:
                continue
            dense_counts = self.count_tables[n].clone()
            dense_counts += 1e-10 # Smoothing
            sums = dense_counts.sum(dim=-1, keepdim=True)
            probs = dense_counts / sums
            self.prob_tables[n] = probs
    def merge_ngram_tables(self, other_count_tables:Dict[int,torch.Tensor]):
        print("merging ngram tables")
        for n in self.n_grams:
            if n in other_count_tables:
                if n in self.count_tables:
                    self.count_tables[n]=self.count_tables[n]+other_count_tables[n]
                else:
                    self.count_tables[n]=other_count_tables[n].clone()
        self.build_prob_tables_from_counts()
        print("ngram tables merged")

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

        #Prepare Ground Truth
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
        kl_per_position, kl_all_values = ngram_analyzer.compute_kl_divergence_batch(model_logits, sequences, n)
        
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