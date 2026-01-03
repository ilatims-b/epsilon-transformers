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
    def __init__(self, hook: str = 'blocks.0.hook_resid_post',device='cpu'):
        self.hook = hook
        self.device = device
        self.test_inputs: Optional[torch.Tensor] = None
        self.test_beliefs: Optional[torch.Tensor] = None
        self.ground_truth_simplex_coords: Optional[np.ndarray] = None

    def setup_from_tree(self, process, depth: int, num_samples: int=None):
        msp_tree = process.derive_mixed_state_presentation(depth=depth)
        
        sequences = []
        beliefs = []
        
        # Collect nodes
        valid_paths = [node.path for node in msp_tree.nodes if len(node.path) == depth-1]
         
        if num_samples is not None and len(valid_paths) > num_samples:
            indices = np.random.choice(len(valid_paths), size=num_samples, replace=False)
            valid_paths = [valid_paths[i] for i in indices]


        self.test_inputs=torch.tensor(valid_paths, dtype=torch.long, device=self.device)
        beliefs_tensor = self._compute_beliefs_for_batch(self.test_inputs, process)
        self.test_beliefs_flat=beliefs_tensor.reshape(-1, beliefs_tensor.shape[-1]).cpu().numpy()

        true_x, true_y = _project_to_simplex(self.test_beliefs_flat)
        self.ground_truth_simplex_coords = np.stack([true_x, true_y], axis=1)

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
        
        return float(mse)    
        
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

    # def _get_tensor(self, data, device):
    #     if not isinstance(data, torch.Tensor):
    #         return torch.tensor(data, dtype=torch.float32, device=device)
    #     return data.to(device=device, dtype=torch.float32)

    # def compute_ground_truth_beliefs(self, sequences: torch.Tensor, process) -> torch.Tensor:
    #     """
    #     Compute the sequence of belief states (mixed states) for the given token sequences.
    #     Replicates the logic of tracking the HMM state distribution.
        
    #     Returns:
    #         Tensor of shape [Batch, Seq_Len, Num_States]
    #     """
    #     batch_size, seq_len = sequences.shape
    #     device = sequences.device
        
    #     # 1. Get Transition Matrices
    #     T_emit = self._get_tensor(process.transition_matrix, device)
    #     if hasattr(process, 'norm_transition_matrix'):
    #         T_next = self._get_tensor(process.norm_transition_matrix, device)
    #     else:
    #         T_next = T_emit

    #     # 2. Initial State (Steady State)
    #     vec = process.steady_state_vector
    #     if isinstance(vec, np.ndarray):
    #         vec = torch.from_numpy(vec).float().to(device)
    #     current_states = vec.unsqueeze(0).expand(batch_size, -1) # [B, S]

    #     all_beliefs = []

    #     # 3. Filter forward
    #     # We need beliefs at each step *before* predicting the next token? 
    #     # Or *after* seeing the token?
    #     # Typically, the model's residual stream at pos `t` contains information 
    #     # relevant for predicting `t+1`. The belief state at `t` (after seeing `0...t`)
    #     # is the predictor for `t+1`.
        
    #     # Initial belief (before any tokens) - usually not mapped by transformers 0-th pos unless BOS
    #     # We will track belief *after* updating with the token at `pos`.
        
    #     for pos in range(seq_len):
    #         emissions = sequences[:, pos] # [B]
            
    #         # Select transition matrices for the observed emissions
    #         T_selected = T_next[emissions] # [B, S, S]
            
    #         # Update state: Next = Current * T
    #         next_states = torch.einsum("bs, bsd -> bd", current_states, T_selected)
            
    #         # Normalize
    #         denom = next_states.sum(dim=1, keepdim=True)
    #         # Avoid division by zero if impossible sequence
    #         denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    #         next_states = next_states / denom
            
    #         current_states = next_states
    #         all_beliefs.append(current_states)
            
    #     return torch.stack(all_beliefs, dim=1) # [B, Seq_Len, S]

    # def compute_simplex_mse(self, 
    #                       model, 
    #                       sequences: torch.Tensor, 
    #                       process, 
    #                       num_samples: Optional[int] = None) -> float:
    #     """
    #     Full pipeline: 
    #     1. Get Activations
    #     2. Get Ground Truth Beliefs
    #     3. Train Probe
    #     4. Calculate MSE of projections
    #     """
    #     device = sequences.device
        
    #     # Subsample if necessary
    #     if num_samples is not None and sequences.shape[0] > num_samples:
    #         indices = torch.randperm(sequences.shape[0])[:num_samples]
    #         sequences = sequences[indices]
            
    #     # 1. Get Activations
    #     # Run with cache to get the hidden states
    #     # We assume the model has been run or we run it here.
    #     # Since we need specific internal activations, we must run it.
    #     with torch.no_grad():
    #         _, cache = model.run_with_cache(sequences, names_filter=lambda x: x == self.hook)
    #         activations = cache[self.hook] # [Batch, Seq_Len, D_Model]

    #     # 2. Get Ground Truth Beliefs
    #     ground_truth_beliefs = self.compute_ground_truth_beliefs(sequences, process)
        
    #     # 3. Flatten
    #     batch_size, n_ctx, d_model = activations.shape
    #     belief_dim = ground_truth_beliefs.shape[-1]
        
    #     activations_flat = activations.reshape(-1, d_model).cpu().numpy()
    #     beliefs_flat = ground_truth_beliefs.reshape(-1, belief_dim).cpu().numpy()
        
    #     # 4. Train Linear Probe (Activations -> Beliefs)
    #     reg = LinearRegression()
    #     reg.fit(activations_flat, beliefs_flat)
    #     beliefs_pred_flat = reg.predict(activations_flat)
        
    #     # 5. Project to Simplex (2D)
    #     # _project_to_simplex returns tuple(x, y)
    #     true_x, true_y = _project_to_simplex(beliefs_flat)
    #     pred_x, pred_y = _project_to_simplex(beliefs_pred_flat)
        
    #     true_2d = np.stack([true_x, true_y], axis=1)
    #     pred_2d = np.stack([pred_x, pred_y], axis=1)
        
    #     # 6. Compute MSE
    #     squared_diffs = (true_2d - pred_2d) ** 2
    #     mse = np.mean(np.sum(squared_diffs, axis=1))
        
    #     return float(mse)
