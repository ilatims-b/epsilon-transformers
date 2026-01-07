# import torch
# import torch.nn.functional as F
# from typing import Dict, Tuple, Optional
# import numpy as np
# from sklearn.linear_model import LinearRegression

# from epsilon_transformers.visualization.plots import _project_to_simplex

# class SimplexAnalyzer:
#     """Calculates the MSE between model's simplex and ground truth simplex
#     For getting simplex, we use _project_to_simplex from visualization.plots
#     """
#     def __init__(self, vocab_size: int, seq_len: Optional[int] = None,hook:str='blocks.0.hook_resid_post', transformer_input_beliefs=None, transformer_inputs=None, model=None):
#         self.vocab_size = vocab_size
#         self.seq_len = seq_len
#         self.hook=hook
#         self.transformer_input_beliefs=transformer_input_beliefs
#         self.transformer_inputs=transformer_inputs
#         self.model=model

#     def run_activation_to_beliefs_regression(self, activations, ground_truth_beliefs):
#         # make sure the first two dimensions are the same
#         assert activations.shape[0] == ground_truth_beliefs.shape[0]
#         assert activations.shape[1] == ground_truth_beliefs.shape[1]
#         # flatten the activations
#         batch_size, n_ctx, d_model = activations.shape
#         belief_dim = ground_truth_beliefs.shape[-1]
#         activations_flattened = activations.view(-1, d_model) # [batch * n_ctx, d_model]
#         ground_truth_beliefs_flattened = ground_truth_beliefs.view(-1, belief_dim) # [batch * n_ctx, belief_dim]
#         # run the regression
#         regression = LinearRegression()
#         regression.fit(activations_flattened, ground_truth_beliefs_flattened)
#         # get the belief predictions
#         belief_predictions = regression.predict(activations_flattened) # [batch * n_ctx, belief_dim]
#         belief_predictions = belief_predictions.reshape(batch_size, n_ctx, belief_dim)

#         return regression, belief_predictions
#     def compute_ground_truth_simplex(self,):
#         transformer_input_belief_flattened=self.transformer_input_beliefs.reshape(-1,3)
#         true_x, true_y = _project_to_simplex(transformer_input_belief_flattened)
#         ground_truth_simplex_coords = np.stack([true_x, true_y], axis=1)
#         return ground_truth_simplex_coords
    
#     def compute_model_simplex_from_activations(self,):
#         _, activations= self.model.run_with_cache(self.transformer_inputs, names_filter=lambda x: self.hook in x)
#         acts=activations[f'{self.hook}']  # [batch, n_ctx, d_model]
#         _, belief_predictions = self.run_activation_to_beliefs_regression(acts, self.transformer_input_beliefs)
#         belief_predictions_flattened=belief_predictions.reshape(-1,3)
#         model_x, model_y = _project_to_simplex(belief_predictions_flattened)
#         model_simplex_coords = np.stack([model_x, model_y], axis=1)
#         return  model_simplex_coords
    
#     def compute_simplex_mse(self, model_simplex_coords, ground_truth_simplex_coords):
#          # Squared Euclidean distance per point: (x1-x2)^2 + (y1-y2)^2
#         squared_diffs = (model_simplex_coords - ground_truth_simplex_coords) ** 2
#         euclidean_squared = np.sum(squared_diffs, axis=1)
#         mse_loss = np.mean(euclidean_squared)
#         return mse_loss
    
# def compute_simplex_mse_for_model(model, transformer_inputs, transformer_input_beliefs, hook='resid_post'):
#     analyzer = SimplexAnalyzer(
#         vocab_size=model.cfg.vocab_size,
#         seq_len=model.cfg.n_ctx,
#         hook=hook,
#         transformer_input_beliefs=transformer_input_beliefs,
#         transformer_inputs=transformer_inputs,
#         model=model
#     )
#     ground_truth_simplex_coords = analyzer.compute_ground_truth_simplex()
#     model_simplex_coords = analyzer.compute_model_simplex_from_activations()
#     mse_loss = analyzer.compute_simplex_mse(model_simplex_coords, ground_truth_simplex_coords)
#     return mse_loss

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from epsilon_transformers.visualization.plots import _project_to_simplex

class SimplexAnalyzer:
    """
    Calculates the MSE between model's internal representation (projected to 2D)
    and the ground truth process beliefs (projected to 2D).
    
    Dynamically computes beliefs from sequences using the process transition matrices.
    """
    def __init__(self, hook: str = 'blocks.0.hook_resid_post',device='cpu', constrained: Optional[bool]=False):
        self.hook = hook
        self.device = device
        self.test_inputs: Optional[torch.Tensor] = None
        self.test_beliefs: Optional[torch.Tensor] = None
        self.ground_truth_simplex_coords: Optional[np.ndarray] = None
        self.constrained: Optional[bool] = False

    def setup_from_tree(self, process, depth: int, num_samples: int=None, constrained: Optional[bool]=False):
        msp_tree = process.derive_mixed_state_presentation(depth=depth)
        
        sequences = []
        beliefs = []
        
        # Collect nodes
        valid_paths = [node.path for node in msp_tree.nodes if len(node.path) == depth-1]
         
        if num_samples is not None and len(valid_paths) > num_samples:
            indices = np.random.choice(len(valid_paths), size=num_samples, replace=False)
            valid_paths = [valid_paths[i] for i in indices]


        self.test_inputs=torch.tensor(valid_paths, dtype=torch.long, device=self.device)
        if constrained:
            beliefs_tensor = self._compute_constrained_beliefs_for_batch(self.test_inputs, process)
        else:
            beliefs_tensor = self._compute_beliefs_for_batch(self.test_inputs, process)
        self.test_beliefs_flat=beliefs_tensor.reshape(-1, beliefs_tensor.shape[-1]).cpu().numpy()

        true_x, true_y = _project_to_simplex(self.test_beliefs_flat)
        self.ground_truth_simplex_coords = np.stack([true_x, true_y], axis=1)
        return beliefs_tensor, self.ground_truth_simplex_coords

    def compute_simplex_mse(self,model) ->float:
        """
        Full pipeline: 
        1. Get Activations
        2. Train Probe
        3. Calculate MSE of projections
        """
        device = self.device
        if self.test_inputs is None:
            raise ValueError("[SimplexAnalyzer] test_inputs not set. Call setup_from_tree() first.")
        model.eval()
        # 1. Get Activations
        with torch.no_grad():
            inputs=self.test_inputs.to(device)
            _, cache = model.run_with_cache(inputs, names_filter=lambda x: x == self.hook)
            activations = cache[self.hook] # [Batch, Seq_Len, D_Model]

        # 2. Flatten
        batch_size, n_ctx, d_model = activations.shape
        
        activations_flat = activations.reshape(-1, d_model).cpu().numpy()
        
        # 3. Train Linear Probe (Activations -> Beliefs)
        reg = LinearRegression()
        reg.fit(activations_flat, self.test_beliefs_flat)
        beliefs_pred_flat = reg.predict(activations_flat)
        
        # 4. Project to Simplex (2D)
        pred_x, pred_y = _project_to_simplex(beliefs_pred_flat)
        
        true_2d = np.stack([self.ground_truth_simplex_coords[:,0], self.ground_truth_simplex_coords[:,1]], axis=1)
        pred_2d = np.stack([pred_x, pred_y], axis=1)
        
        # 5. Compute MSE
        squared_diffs = (true_2d - pred_2d) ** 2
        mse = np.mean(np.sum(squared_diffs, axis=1))
        
        return pred_x, pred_y, float(mse)    
        
    def _compute_beliefs_for_batch(self, sequences, process):
        """Vectorized HMM Filter handling both standard and normalized transitions."""
        batch, seq_len = sequences.shape
        device = sequences.device
        
        # Helper to get matrices on device
        def to_tensor(x): return torch.tensor(x, dtype=torch.float32, device=device)
        
        # CHECK: Use norm_transition_matrix if it exists (for mixed-state processes)
        if hasattr(process, 'norm_transition_matrix'):
            # This matrix is designed for belief updates: Pr(Next State | Current State, Emission)
            # Shape: [Emission, Source_State, Dest_State]
            T = to_tensor(process.norm_transition_matrix)
        else:
            # Fallback for standard HMMs
            T = to_tensor(process.transition_matrix)

        # Initial State (Steady State)
        current = to_tensor(process.steady_state_vector).unsqueeze(0).expand(batch, -1) # [B, S]
        
        all_beliefs = []
        for t in range(seq_len):
            emissions = sequences[:, t] # [B]
            
            # Select correct transition matrices for this step's emissions
            # T_step shape: [Batch, Src, Dst]
            T_step = T[emissions]
            # Update Beliefs: Belief_curr @ T_step
            # [B, 1, S] @ [B, S, S] -> [B, 1, S] -> [B, S]
            next_s = torch.bmm(current.unsqueeze(1), T_step).squeeze(1)
            
            # Normalize to ensure it remains a valid probability distribution
            # (Essential because T_emit usually doesn't sum to 1 across next states for a *specific* emission)
            denom = next_s.sum(dim=1, keepdim=True)
            # Avoid division by zero for impossible transitions (numerical stability)
            denom = torch.where(denom < 1e-12, torch.ones_like(denom), denom)
            next_s = next_s / denom
            
            current = next_s
            all_beliefs.append(current)
            
        return torch.stack(all_beliefs, dim=1)
    
    def _compute_constrained_beliefs_for_batch(self, sequences, process):
        """
        Computes constrained beliefs: 
        B_d = pi + sum_{n=0}^{d-1} (pi @ T^{|z_{d-n}} @ T^n - pi)
        
        Where n is the 'lag' (0 for the most recent token, d-1 for the first token).
        """
        batch, seq_len = sequences.shape
        device = sequences.device
        num_states = process.num_states
        pi = torch.tensor(process.steady_state_vector, dtype=torch.float32, device=device) # [S]
        T_labeled = torch.tensor(process.transition_matrix, dtype=torch.float32, device=device)
        if T_labeled.ndim == 3:
            T_std = T_labeled.sum(dim=0) # [S, S]
        else:
            raise ValueError(f"Expected process.transition_matrix to be [Vocab, S, S], got {T_labeled.shape}")
        if hasattr(process, 'norm_transition_matrix'):
             T_norm = torch.tensor(process.norm_transition_matrix, dtype=torch.float32, device=device)
        else:
             row_sums = T_labeled.sum(dim=2, keepdim=True) # [Vocab, S, 1]
             safe_row_sums = torch.where(row_sums < 1e-12, torch.ones_like(row_sums), row_sums)
             T_norm = T_labeled / safe_row_sums

        # DIAGNOSTIC: Check if matrices are valid
        print(f"Pi sum: {pi.sum().item():.6f}")
        print(f"T_std row sums: {T_std.sum(dim=1)}")
        print(f"T_norm row sums: {T_norm.sum(dim=2)}")     
            # 2. Precompute powers of T: T^0, T^1, ..., T^{L-1}
        T_powers = [torch.eye(num_states, device=device)]
        curr_T = T_std
        for _ in range(seq_len): 
            T_powers.append(curr_T)
            curr_T = torch.mm(curr_T, T_std)
        T_powers = torch.stack(T_powers) # [L+1, S, S]
        #  Compute "Base Perturbations" (pi @ T^{|z}) for every token in the sequenc
        flat_seqs = sequences.flatten() # [B*L]
        T_selected = T_norm[flat_seqs] # [B*L, S, S]
        # Expand pi: [B*L, 1, S]
        pi_expanded = pi.unsqueeze(0).unsqueeze(1).expand(batch * seq_len, -1, -1)
        # Compute pi @ T^{|z}: [B*L, 1, S]
        # This is the "excited" state immediately after seeing token z
        pi_excited_flat = torch.bmm(pi_expanded, T_selected).squeeze(1) # [B*L, S]
        pi_excited = pi_excited_flat.view(batch, seq_len, num_states) # [B, L, S]
        # Compute belief at each step d (where d is 1-indexed length in formula, 0-indexed t here)
        constrained_beliefs = []
        
        for t in range(seq_len):
            # We are computing belief at index t (length t+1).
            # The formula is sum_{n=0}^{t} (pi @ T^{|z_{t-n}} @ T^n - pi)
            
            # Slice relevant excited states: [B, t+1, S]
            # These correspond to tokens z_0, ..., z_t
            excited_slice = pi_excited[:, :t+1, :] 
            
            # We need to match z_k with T^{t-k}. 
            # z_t (index t) gets T^0. z_0 (index 0) gets T^t.
            # So if excited_slice is [z_0, z_1, ..., z_t], we need T_powers [T^t, T^{t-1}, ..., T^0]
            # Indices for T_powers: [t, t-1, ..., 0]
            power_indices = torch.arange(t, -1, -1, device=device)
            T_slice = T_powers[power_indices] # [t+1, S, S]
            
            # Compute (pi @ T^{|z}) @ T^n
            # excited_slice[b, k] is (pi @ T^{|z_k})
            # T_slice[k] is T^{t-k}
            # term_k = excited_slice[b, k] @ T_slice[k]
            propagated_terms = torch.einsum('bks,ksd->bkd', excited_slice, T_slice) # [B, t+1, S]
            
            # (pi @ T^{|z} @ T^n) - pi
            # Broadcasting pi: [S] -> [1, 1, S]
            terms_minus_pi = propagated_terms - pi.view(1, 1, -1)
            
            # Sum all terms
            sum_terms = terms_minus_pi.sum(dim=1) # [B, S]
            
            #  Add baseline pi
            belief_t = pi.unsqueeze(0) + sum_terms
            # DIAGNOSTIC: Check validity
            print(f"t={t}: belief_t sum={belief_t[0].sum().item():.6f}, min={belief_t.min().item():.6f}, max={belief_t.max().item():.6f}")
            constrained_beliefs.append(belief_t)
            
        return torch.stack(constrained_beliefs, dim=1)