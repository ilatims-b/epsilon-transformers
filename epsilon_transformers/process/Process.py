import numpy as np
from typing import Iterator, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float
from collections import deque
import torch

from epsilon_transformers.process.MixedStateTree import (
    MixedStateTree,
    MixedStateTreeNode,
)

# TODO: Test yield_emission_histories for different emissions in the emission history
# TODO: Rename _create_hmm
# TODO: Delete generate_process_history (??)


@dataclass
class ProcessHistory:
    symbols: list[int]
    states: list[str]

    def __post_init__(self):
        pass
        # assert len(self.symbols) == len(
        #     self.states
        # ), "length of symbols & states must be the same"

    def __len__(self):
        return len(self.states)

    
class Process(ABC):
    name: str
    transition_matrix: Float[np.ndarray, "vocab_len num_states num_states"]
    state_names_dict: dict[str, int]
    vocab_len: int
    num_states: int
    steady_state_vector: Float[np.ndarray, "num_states"]
    vocab_map: Optional[Union[list[int], dict[int, int]]]
    
    # GPU tensors (lazily initialized)
    _gpu_transition_matrix: Optional[torch.Tensor] = None
    _gpu_steady_state: Optional[torch.Tensor] = None
    _gpu_vocab_map: Optional[torch.Tensor] = None
    _gpu_device: Optional[torch.device] = None

    def __init__(self, vocab_map: Optional[Union[list[int], dict[int, int]]] = None):
        self.transition_matrix, self.state_names_dict = self._create_hmm()
        
        if (
            len(self.transition_matrix.shape) != 3
            or self.transition_matrix.shape[1] != self.transition_matrix.shape[2]
        ):
            raise ValueError(
                "Transition matrix should have 3 axes and the final two dims shoulds be square"
            )

        if self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should be square")

        transition = self.transition_matrix.sum(axis=0)
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("Transition matrix should be stochastic and sum to 1")

        self.vocab_len = self.transition_matrix.shape[0]
        self.num_states = self.transition_matrix.shape[1]
        self.steady_state_vector = self._compute_steady_state()
        
        # Vocabulary Mapping Setup
        self.vocab_map = vocab_map
        if self.vocab_map is not None:
            if len(self.vocab_map) != self.vocab_len:
                raise ValueError(f"vocab_map length ({len(self.vocab_map)}) must match process vocab_len ({self.vocab_len})")
        
        # Reset GPU cache
        self._gpu_transition_matrix = None
        self._gpu_steady_state = None
        self._gpu_vocab_map = None
        self._gpu_device = None

    def _compute_steady_state(self) -> Float[np.ndarray, "num_states"]:
        """Calculates steady state vector from transition matrix."""
        state_transition_matrix = np.sum(self.transition_matrix, axis=0)
        eigenvalues, eigenvectors = np.linalg.eig(state_transition_matrix.T)
        steady_state_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        
        if steady_state_vector.shape[1] > 0:
            steady_state_vector = steady_state_vector[:, 0]
            
        normalized_steady_state_vector = steady_state_vector / steady_state_vector.sum()
        return normalized_steady_state_vector

    @property
    def is_unifilar(self) -> bool:
        for i in range(self.num_states):
            for j in range(self.vocab_len):
                if np.count_nonzero(self.transition_matrix[j, i, :]) > 1:
                    return False
        return True

    @abstractmethod
    def _create_hmm(
        self,
    ) -> tuple[Float[np.ndarray, "vocab_len num_states num_states"], dict[str, int]]:
        ...

    def _apply_vocab_map(self, emission_idx: int) -> int:
        if self.vocab_map is None:
            return emission_idx
        if isinstance(self.vocab_map, dict):
            return self.vocab_map[emission_idx]
        return self.vocab_map[emission_idx]

    def _ensure_gpu_tensors(self, device: torch.device):
        """Lazily initialize GPU tensors for batch generation."""
        if self._gpu_device != device or self._gpu_transition_matrix is None:
            self._gpu_transition_matrix = torch.tensor(
                self.transition_matrix, dtype=torch.float32, device=device
            )
            if self.vocab_map is not None:
                # Convert dict to list for tensor creation if necessary
                v_map = self.vocab_map
                if isinstance(v_map, dict):
                    v_map = [v_map[i] for i in range(self.vocab_len)]
                self._gpu_vocab_map = torch.tensor(v_map, dtype=torch.long, device=device)
            
            self._gpu_device = device

    def _get_gpu_steady_state(self, device: torch.device) -> torch.Tensor:
        if self._gpu_steady_state is None or self._gpu_device != device:
            self._gpu_steady_state = torch.tensor(
                self.steady_state_vector, dtype=torch.float32, device=device
            )
        return self._gpu_steady_state

    def generate_batch_gpu(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
    ) -> torch.Tensor:
        self._ensure_gpu_tensors(device)
        T = self._gpu_transition_matrix 
        
        if start_state_idx is not None:
            current_states = torch.full((batch_size,), start_state_idx, dtype=torch.long, device=device)
        else:
            steady_state = self._get_gpu_steady_state(device)
            probs = steady_state.unsqueeze(0).expand(batch_size, -1) if steady_state.ndim==1 else steady_state
            current_states = torch.multinomial(probs, num_samples=1).squeeze(-1)

        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        
        for t in range(seq_len):
            trans_probs = T[:, current_states, :].permute(1, 0, 2)
            flat_probs = trans_probs.reshape(batch_size, -1)
            joint_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            
            emission = joint_idx // self.num_states
            next_state = joint_idx % self.num_states
            
            emissions[:, t] = emission
            current_states = next_state
        
        # Apply vocab mapping if it exists
        if self._gpu_vocab_map is not None:
            emissions = self._gpu_vocab_map[emissions]
            
        return emissions
        
    def generate_batch_gpu_with_beliefs(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            emissions: (batch_size, seq_len) - Discrete observations
            true_states: (batch_size, seq_len) - Ground truth state indices
            beliefs: (batch_size, seq_len, num_states) - Observer's belief distributions
        """
        self._ensure_gpu_tensors(device)
        T = self._gpu_transition_matrix  # (vocab_len, num_states, num_states)
        
        # 1. Initialize Ground Truth States
        if start_state_idx is not None:
            current_states = torch.full((batch_size,), start_state_idx, dtype=torch.long, device=device)
            # Initial belief is certain
            current_belief = torch.zeros((batch_size, self.num_states), device=device)
            current_belief[:, start_state_idx] = 1.0
        else:
            steady_state = self._get_gpu_steady_state(device)
            current_states = torch.multinomial(
                steady_state.unsqueeze(0).expand(batch_size, -1), 
                num_samples=1
            ).squeeze(-1)
            # Initial belief starts at steady state
            current_belief = steady_state.unsqueeze(0).expand(batch_size, -1)

        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        true_states = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        beliefs = torch.empty(batch_size, seq_len, self.num_states, dtype=torch.float32, device=device)

        for t in range(seq_len):
            # --- A. Generate Next Step (Ground Truth) ---
            trans_probs = T[:, current_states, :].permute(1, 0, 2) # (batch_size, vocab_len, num_states)
            flat_probs = trans_probs.reshape(batch_size, -1)
            
            joint_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            
            emission = joint_idx // self.num_states
            next_state = joint_idx % self.num_states
            
            # --- B. Update Observer's Belief (Filtering) ---
            # The observer sees 'emission' and updates current_belief
            # T_emit shape: (batch_size, num_states, num_states)
            T_emit = T[emission] 
            
            # Belief Update: b_{t} = (b_{t-1} @ T_emit) / Normalization
            # [B, 1, S] @ [B, S, S] -> [B, 1, S] -> [B, S]
            next_belief = torch.bmm(current_belief.unsqueeze(1), T_emit).squeeze(1)
            
            # Normalize
            denom = next_belief.sum(dim=1, keepdim=True)
            current_belief = next_belief / torch.where(denom < 1e-12, torch.ones_like(denom), denom)

            if self._gpu_vocab_map is not None:
                stored_emission=self._gpu_vocab_map[emission]
            else:
                stored_emission=emission    
            # --- C. Store results ---
            emissions[:, t] = stored_emission
            true_states[:, t] = next_state
            beliefs[:, t] = current_belief
            
            current_states = next_state
        
        return emissions, true_states, beliefs

    def _sample_emission(self, current_state_idx: Optional[int] = None) -> int:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )

        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"

        p = self.transition_matrix[:, current_state_idx, :].sum(axis=1)
        emission = np.random.choice(self.vocab_len, p=p)
        if self.vocab_map is not None:
            emission = self._apply_vocab_map(emission)
        return emission    

    def yield_emissions(
        self, sequence_len: int, current_state_idx: int | None = None
    ) -> Iterator[int]:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )
        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"
        for _ in range(sequence_len):
            emission, next_state_idx = self._sample_emission_and_next_state(
                current_state_idx
            )
            yield emission
            current_state_idx = next_state_idx


    def _sample_emission_and_next_state(
        self, current_state_idx: int
    ) -> tuple[int, int]:
        transition_probs = self.transition_matrix[:, current_state_idx, :] 
        emission_next_state_idx = np.random.choice(
            transition_probs.size, p=transition_probs.ravel()
        )
        emission = emission_next_state_idx // self.num_states
        next_state_idx = emission_next_state_idx % self.num_states
        if self.vocab_map is not None:
            emission = self._apply_vocab_map(emission)
        return emission, next_state_idx
    
    def yield_emission_histories(
        self, sequence_len: int, num_sequences: int, start_state_idx: Optional[int]=None
    ) -> Iterator[list[int]]:
        for _ in range(num_sequences):
            yield [x for x in self.yield_emissions(sequence_len=sequence_len, current_state_idx=start_state_idx)]

    def generate_process_history(
        self, total_length: int, current_state_idx: int | None = None
    ) -> ProcessHistory:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )
        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"

        index_to_state_names_dict = {v: k for k, v in self.state_names_dict.items()}

        symbols = []
        states = []

        for _ in range(total_length):
            states.append(index_to_state_names_dict[current_state_idx])
            emission, next_state_idx = self._sample_emission_and_next_state(
                current_state_idx
            )
            symbols.append(emission)
            current_state_idx = next_state_idx

        return ProcessHistory(symbols=symbols, states=states)

    # TODO: You can get rid of the stack, and just iterate through the nodes & the depth as tuples
    #added
    def derive_mixed_state_presentation(self, depth: int,  start_state_idx: Optional[int] = None) -> MixedStateTree:
        if start_state_idx is not None:
            assert 0 <= start_state_idx < self.num_states, "start_state_idx out of bounds"
            initial_dist = np.zeros(self.num_states)
            initial_dist[start_state_idx] = 1.0
        else:
            initial_dist = self.steady_state_vector

        tree_root = MixedStateTreeNode(
            state_prob_vector = initial_dist,
            children=set(),
            path=[],
            emission_prob=0,
        )
        nodes = set([tree_root])

        stack: deque[
            tuple[MixedStateTreeNode, Float[np.ndarray, "num_states"], list[int], int]
        ] = deque([(tree_root, initial_dist, [], 0)])
        while stack:
            current_node, state_prob_vector, current_path, current_depth = stack.pop()
            if current_depth < depth:
                emission_probs = _compute_emission_probabilities(
                    self, state_prob_vector
                )
                for emission in range(self.vocab_len):
                    if emission_probs[emission] > 0:
                        next_state_prob_vector = _compute_next_distribution(
                            self.transition_matrix, state_prob_vector, emission
                        )
                        child_path = current_path + [emission]
                        child_node = MixedStateTreeNode(
                            state_prob_vector=next_state_prob_vector,
                            path=child_path,
                            children=set(),
                            emission_prob=emission_probs[emission],
                        )
                        current_node.add_child(child_node)

                        stack.append(
                            (
                                child_node,
                                next_state_prob_vector,
                                child_path,
                                current_depth + 1,
                            )
                        )
            nodes.add(current_node)

        return MixedStateTree(
            root_node=tree_root, process=self.name, nodes=nodes, depth=depth
        )

def _compute_emission_probabilities(
    hmm: Process, state_prob_vector: Float[np.ndarray, "num_states"]
) -> Float[np.ndarray, "vocab_len"]:
    """
    Compute the probabilities associated with each emission given the current mixed state.
    """
    T = hmm.transition_matrix
    emission_probs = np.einsum("s,esd->ed", state_prob_vector, T).sum(axis=1)
    emission_probs /= emission_probs.sum()
    return emission_probs

def _compute_next_distribution(
    epsilon_machine: Float[np.ndarray, "vocab_len num_states num_states"],
    current_state_prob_vector: Float[np.ndarray, "num_states"],
    current_emission: int,
) -> Float[np.ndarray, "num_states"]:
    """
    Compute the next mixed state distribution for a given output.
    """
    X_next = np.einsum(
        "sd, s -> d", epsilon_machine[current_emission], current_state_prob_vector
    )
    return X_next / np.sum(X_next) if np.sum(X_next) != 0 else X_next

class MixedProcess:
    """
    A meta-process that switches between multiple sub-processes based on a schedule.
    """
    def __init__(
        self,
        processes: list[Process],
        switch_times: list[int],
        switch_prob: Union[float,list[float]],
        state_mode: str = 'steady', # 'same', 'resume', 'steady'
        vocab_map: Optional[Union[list[int], dict[int, int]]] = None
    ):
        self.processes = processes
        if isinstance(switch_prob, float):
            self.switch_schedule={t: switch_prob for t in switch_times}
        elif isinstance(switch_prob, list):
            if len(switch_prob) != len(switch_times):
                raise ValueError("If switch_prob is a list, it must have the same length as switch_times")
            self.switch_schedule={t: p for t, p in zip(switch_times, switch_prob)}
        else:
            raise ValueError("switch_prob must be either a float or a list of floats")
                
        # self.switch_times = set(switch_times)
        self.state_mode = state_mode
        self.num_processes = len(processes)
        self.vocab_map=vocab_map
        
        if self.state_mode == 'same':
            # Verify all processes have compatible state spaces
            n_states = processes[0].num_states
            for p in processes:
                if p.num_states != n_states:
                    raise ValueError("All processes must have the same number of states for 'same' switching mode.")

    def generate_batch_gpu(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generates a sequence by mixing sub-processes.
        """
        # Ensure all sub-processes have GPU tensors ready
        for p in self.processes:
            p._ensure_gpu_tensors(device)

        # 1. Initialize State Tracking
        # active_process_indices: which process is currently active for each batch element (0 to num_processes-1)
        active_process_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # current_state_indices: holding current state index for the *active* process
        # We also need storage for 'resume' mode to remember where each process left off
        stored_states = torch.zeros((self.num_processes, batch_size), dtype=torch.long, device=device)
        
        # Initialize states
        # For process 0, use start_state_idx if provided, else steady state
        if start_state_idx is not None:
             stored_states[0] = start_state_idx
        else:
             steady = self.processes[0]._get_gpu_steady_state(device)
             stored_states[0] = torch.multinomial(steady.unsqueeze(0).expand(batch_size, -1), 1).squeeze(-1)
             
        # Initialize other processes to their steady states (needed for 'resume' or initial switch)
        for i in range(1, self.num_processes):
             steady = self.processes[i]._get_gpu_steady_state(device)
             stored_states[i] = torch.multinomial(steady.unsqueeze(0).expand(batch_size, -1), 1).squeeze(-1)
             
        # Initialize current states from stored_states based on active process (initially 0)
        current_states = stored_states[0].clone()

        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        
        # 2. Generation Loop
        for t in range(seq_len):
            # A. Handle Switching
            if t in self.switch_schedule:
                # Decide who switches: Bernoulli(switch_prob)
                current_switch_prob = self.switch_schedule[t]
                should_switch = torch.bernoulli(torch.full((batch_size,), current_switch_prob, device=device)).bool()
                
                if should_switch.any():
                    # Save current states before switching (crucial for 'resume')
                    if self.state_mode == 'resume':
                        # We need to scatter current_states back to stored_states
                        # stored_states[active_process, batch_idx] = current_state
                        # This is tricky to vectorize efficiently without advanced indexing
                        # Simple approach: iterate unique active processes
                        for p_idx in range(self.num_processes):
                            mask = (active_process_indices == p_idx) & should_switch
                            if mask.any():
                                stored_states[p_idx, mask] = current_states[mask]

                    # Update active process index (cyclic switch: 0 -> 1 -> ... -> 0)
                    new_indices = (active_process_indices + 1) % self.num_processes
                    active_process_indices = torch.where(should_switch, new_indices, active_process_indices)
                    
                    # Load new states based on mode
                    mask_switch = should_switch
                    
                    if self.state_mode == 'resume':
                        # Gather stored states for the new process
                        # We iterate to gather because stored_states is (num_proc, batch)
                        for p_idx in range(self.num_processes):
                            mask = (active_process_indices == p_idx) & mask_switch
                            if mask.any():
                                current_states[mask] = stored_states[p_idx, mask]
                                
                    elif self.state_mode == 'steady':
                        # Sample from steady state of the NEW process
                        for p_idx in range(self.num_processes):
                            mask = (active_process_indices == p_idx) & mask_switch
                            if mask.any():
                                steady = self.processes[p_idx]._get_gpu_steady_state(device)
                                # Expand steady to match number of switching elements
                                count = mask.sum().item()
                                if count > 0:
                                    new_samples = torch.multinomial(steady.unsqueeze(0).expand(count, -1), 1).squeeze(-1)
                                    current_states[mask] = new_samples
                                    
                    elif self.state_mode == 'same':
                        # Keep current_state value, just apply to new process dynamics
                        # No update needed to current_states tensor
                        pass

            # B. Generate Step (Vectorized by Process Group)
            # Since different processes have different T matrices, we process them in groups
            step_emissions = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for p_idx, process in enumerate(self.processes):
                mask = (active_process_indices == p_idx)
                if not mask.any():
                    continue
                
                # Extract states for this group
                group_states = current_states[mask]
                group_size = group_states.shape[0]
                
                T = process._gpu_transition_matrix
                
                # Sampling logic (similar to Process.generate_batch_gpu)
                # T: (vocab, state, state)
                # trans_probs: (vocab, group_size, state)
                trans_probs = T[:, group_states, :].permute(1, 0, 2)
                flat_probs = trans_probs.reshape(group_size, -1)
                
                joint_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
                
                emission = joint_idx // process.num_states
                next_state = joint_idx % process.num_states
                
                # Apply vocab map immediately
                if process._gpu_vocab_map is not None:
                    final_emission = process._gpu_vocab_map[emission]
                else:
                    final_emission = emission

                # Store back
                step_emissions[mask] = final_emission
                current_states[mask] = next_state
            
            emissions[:, t] = step_emissions

        return emissions
    def generate_batch_gpu_with_beliefs(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate sequence with beliefs for MixedProcess (state_mode='same' or 'resume').
        """
        if self.state_mode not in ['same', 'resume']:
             raise NotImplementedError(f"generate_batch_gpu_with_beliefs does not support state_mode='{self.state_mode}' yet.")
        
        # Ensure all sub-processes have GPU tensors ready
        for p in self.processes:
            p._ensure_gpu_tensors(device)

        # 1. Initialize State & Belief Tracking
        active_process_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        num_states = self.processes[0].num_states
        
        if self.state_mode == 'same':
            # Single shared reality: One tracker for all processes
            if start_state_idx is not None:
                current_states = torch.full((batch_size,), start_state_idx, dtype=torch.long, device=device)
                current_belief = torch.zeros((batch_size, num_states), device=device)
                current_belief[:, start_state_idx] = 1.0
            else:
                steady = self.processes[0]._get_gpu_steady_state(device)
                current_states = torch.multinomial(steady.unsqueeze(0).expand(batch_size, -1), 1).squeeze(-1)
                current_belief = steady.unsqueeze(0).expand(batch_size, -1).clone()

        elif self.state_mode == 'resume':
            # Parallel universes: Independent trackers for each process
            stored_states = torch.zeros((self.num_processes, batch_size), dtype=torch.long, device=device)
            stored_beliefs = torch.zeros((self.num_processes, batch_size, num_states), dtype=torch.float32, device=device)
            
            for p_idx, p in enumerate(self.processes):
                if p_idx == 0 and start_state_idx is not None:
                    stored_states[p_idx] = start_state_idx
                    stored_beliefs[p_idx, :, start_state_idx] = 1.0
                else:
                    steady = p._get_gpu_steady_state(device)
                    stored_states[p_idx] = torch.multinomial(steady.unsqueeze(0).expand(batch_size, -1), 1).squeeze(-1)
                    stored_beliefs[p_idx] = steady.unsqueeze(0).expand(batch_size, -1).clone()

        # Output Tensors
        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        true_states = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        beliefs = torch.empty(batch_size, seq_len, num_states, dtype=torch.float32, device=device)

        # 2. Generation Loop
        for t in range(seq_len):
            # A. Handle Switching
            if t in self.switch_schedule:
                current_switch_prob = self.switch_schedule[t]
                should_switch = torch.bernoulli(torch.full((batch_size,), current_switch_prob, device=device)).bool()
                
                if should_switch.any():
                    # Update active process index (cyclic switch: 0 -> 1 -> ... -> 0)
                    new_indices = (active_process_indices + 1) % self.num_processes
                    active_process_indices = torch.where(should_switch, new_indices, active_process_indices)

            # B. Generate Step (Vectorized by Process Group)
            step_emissions = torch.zeros(batch_size, dtype=torch.long, device=device)
            step_true_states = torch.zeros(batch_size, dtype=torch.long, device=device)
            step_beliefs = torch.zeros(batch_size, num_states, dtype=torch.float32, device=device)

            for p_idx, process in enumerate(self.processes):
                mask = (active_process_indices == p_idx)
                if not mask.any():
                    continue
                
                # Extract states and beliefs for this group based on state_mode
                if self.state_mode == 'same':
                    group_states = current_states[mask]
                    group_belief = current_belief[mask]
                else: # 'resume'
                    group_states = stored_states[p_idx, mask]
                    group_belief = stored_beliefs[p_idx, mask]
                
                group_size = group_states.shape[0]
                T = process._gpu_transition_matrix
                
                # --- Generation (Ground Truth) ---
                trans_probs = T[:, group_states, :].permute(1, 0, 2)
                flat_probs = trans_probs.reshape(group_size, -1)
                joint_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
                
                emission = joint_idx // process.num_states
                next_state = joint_idx % process.num_states
                
                # --- Filtering (Belief Update) ---
                T_emit = T[emission] # (group_size, num_states, num_states)
                
                # Belief Update: b_{t} = (b_{t-1} @ T_emit) / Normalization
                next_belief = torch.bmm(group_belief.unsqueeze(1), T_emit).squeeze(1)
                denom = next_belief.sum(dim=1, keepdim=True)
                group_belief = next_belief / torch.where(denom < 1e-12, torch.ones_like(denom), denom)

                # --- Store back ---
                if process._gpu_vocab_map is not None:
                    final_emission = process._gpu_vocab_map[emission]
                else:
                    final_emission = emission

                step_emissions[mask] = final_emission
                step_true_states[mask] = next_state
                step_beliefs[mask] = group_belief
                
                # Update the specific tracking mechanism
                if self.state_mode == 'same':
                    current_states[mask] = next_state
                    current_belief[mask] = group_belief
                else: # 'resume'
                    stored_states[p_idx, mask] = next_state
                    stored_beliefs[p_idx, mask] = group_belief
            
            emissions[:, t] = step_emissions
            true_states[:, t] = step_true_states
            beliefs[:, t] = step_beliefs

        return emissions, true_states, beliefs
    from typing import Optional

    def generate_batch_gpu_with_beliefs_proc_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate sequence with beliefs for MixedProcess (state_mode='same' or 'resume').
        Returns:
            emissions: (batch_size, seq_len)
            true_states: (batch_size, seq_len)
            beliefs: (batch_size, seq_len, num_states)
            process_masks: (batch_size, seq_len) - The ID of the active process at each step.
        """
        if self.state_mode not in ['same', 'resume']:
             raise NotImplementedError(f"generate_batch_gpu_with_beliefs does not support state_mode='{self.state_mode}' yet.")
        
        # Ensure all sub-processes have GPU tensors ready
        for p in self.processes:
            p._ensure_gpu_tensors(device)

        # 1. Initialize State & Belief Tracking
        active_process_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        num_states = self.processes[0].num_states
        
        if self.state_mode == 'same':
            # Single shared reality: One tracker for all processes
            if start_state_idx is not None:
                current_states = torch.full((batch_size,), start_state_idx, dtype=torch.long, device=device)
                current_belief = torch.zeros((batch_size, num_states), device=device)
                current_belief[:, start_state_idx] = 1.0
            else:
                steady = self.processes[0]._get_gpu_steady_state(device)
                current_states = torch.multinomial(steady.unsqueeze(0).expand(batch_size, -1), 1).squeeze(-1)
                current_belief = steady.unsqueeze(0).expand(batch_size, -1).clone()

        elif self.state_mode == 'resume':
            # Parallel universes: Independent trackers for each process
            stored_states = torch.zeros((self.num_processes, batch_size), dtype=torch.long, device=device)
            stored_beliefs = torch.zeros((self.num_processes, batch_size, num_states), dtype=torch.float32, device=device)
            
            for p_idx, p in enumerate(self.processes):
                if p_idx == 0 and start_state_idx is not None:
                    stored_states[p_idx] = start_state_idx
                    stored_beliefs[p_idx, :, start_state_idx] = 1.0
                else:
                    steady = p._get_gpu_steady_state(device)
                    stored_states[p_idx] = torch.multinomial(steady.unsqueeze(0).expand(batch_size, -1), 1).squeeze(-1)
                    stored_beliefs[p_idx] = steady.unsqueeze(0).expand(batch_size, -1).clone()

        # Output Tensors
        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        true_states = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        beliefs = torch.empty(batch_size, seq_len, num_states, dtype=torch.float32, device=device)
        
        # NEW: Tensor to track the active process ID
        process_masks = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)

        # 2. Generation Loop
        for t in range(seq_len):
            # A. Handle Switching
            if t in self.switch_schedule:
                current_switch_prob = self.switch_schedule[t]
                should_switch = torch.bernoulli(torch.full((batch_size,), current_switch_prob, device=device)).bool()
                
                if should_switch.any():
                    # Update active process index (cyclic switch: 0 -> 1 -> ... -> 0)
                    new_indices = (active_process_indices + 1) % self.num_processes
                    active_process_indices = torch.where(should_switch, new_indices, active_process_indices)

            # B. Generate Step (Vectorized by Process Group)
            step_emissions = torch.zeros(batch_size, dtype=torch.long, device=device)
            step_true_states = torch.zeros(batch_size, dtype=torch.long, device=device)
            step_beliefs = torch.zeros(batch_size, num_states, dtype=torch.float32, device=device)

            for p_idx, process in enumerate(self.processes):
                mask = (active_process_indices == p_idx)
                if not mask.any():
                    continue
                
                # Extract states and beliefs for this group based on state_mode
                if self.state_mode == 'same':
                    group_states = current_states[mask]
                    group_belief = current_belief[mask]
                else: # 'resume'
                    group_states = stored_states[p_idx, mask]
                    group_belief = stored_beliefs[p_idx, mask]
                
                group_size = group_states.shape[0]
                T = process._gpu_transition_matrix
                
                # --- Generation (Ground Truth) ---
                trans_probs = T[:, group_states, :].permute(1, 0, 2)
                flat_probs = trans_probs.reshape(group_size, -1)
                joint_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
                
                emission = joint_idx // process.num_states
                next_state = joint_idx % process.num_states
                
                # --- Filtering (Belief Update) ---
                T_emit = T[emission] # (group_size, num_states, num_states)
                
                # Belief Update: b_{t} = (b_{t-1} @ T_emit) / Normalization
                next_belief = torch.bmm(group_belief.unsqueeze(1), T_emit).squeeze(1)
                denom = next_belief.sum(dim=1, keepdim=True)
                group_belief = next_belief / torch.where(denom < 1e-12, torch.ones_like(denom), denom)

                # --- Store back ---
                if process._gpu_vocab_map is not None:
                    final_emission = process._gpu_vocab_map[emission]
                else:
                    final_emission = emission

                step_emissions[mask] = final_emission
                step_true_states[mask] = next_state
                step_beliefs[mask] = group_belief
                
                # Update the specific tracking mechanism
                if self.state_mode == 'same':
                    current_states[mask] = next_state
                    current_belief[mask] = group_belief
                else: # 'resume'
                    stored_states[p_idx, mask] = next_state
                    stored_beliefs[p_idx, mask] = group_belief
            
            # Record everything for timestep t
            emissions[:, t] = step_emissions
            true_states[:, t] = step_true_states
            beliefs[:, t] = step_beliefs
            process_masks[:, t] = active_process_indices

        return emissions, true_states, beliefs, process_masks
    # def generate_batch_gpu_with_beliefs(
    #     self,
    #     batch_size: int,
    #     seq_len: int,
    #     device: torch.device,
    #     start_state_idx: Optional[int] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Generate sequence with beliefs for MixedProcess (state_mode='same').
    #     """
    #     if self.state_mode != 'same':
    #          raise NotImplementedError("generate_batch_gpu_with_beliefs only supports state_mode='same' for now.")
        
    #     # Ensure all sub-processes have GPU tensors ready
    #     for p in self.processes:
    #         p._ensure_gpu_tensors(device)

    #     # 1. Initialize State Tracking
    #     active_process_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
    #     # In state_mode='same', all processes share the state index
    #     if start_state_idx is not None:
    #         current_states = torch.full((batch_size,), start_state_idx, dtype=torch.long, device=device)
    #         # Initial belief is certain
    #         current_belief = torch.zeros((batch_size, self.processes[0].num_states), device=device)
    #         current_belief[:, start_state_idx] = 1.0
    #     else:
    #         steady = self.processes[0]._get_gpu_steady_state(device)
    #         current_states = torch.multinomial(steady.unsqueeze(0).expand(batch_size, -1), 1).squeeze(-1)
    #          # Initial belief starts at steady state
    #         current_belief = steady.unsqueeze(0).expand(batch_size, -1)

    #     emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    #     true_states = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    #     beliefs = torch.empty(batch_size, seq_len, self.processes[0].num_states, dtype=torch.float32, device=device)

        
    #     # 2. Generation Loop
    #     for t in range(seq_len):
    #         # A. Handle Switching
    #         if t in self.switch_schedule:
    #             current_switch_prob = self.switch_schedule[t]
    #             should_switch = torch.bernoulli(torch.full((batch_size,), current_switch_prob, device=device)).bool()
                
    #             if should_switch.any():
    #                 # Update active process index (cyclic switch: 0 -> 1 -> ... -> 0)
    #                 new_indices = (active_process_indices + 1) % self.num_processes
    #                 active_process_indices = torch.where(should_switch, new_indices, active_process_indices)

    #         # B. Generate Step (Vectorized by Process Group)
    #         step_emissions = torch.zeros(batch_size, dtype=torch.long, device=device)
    #         step_true_states = torch.zeros(batch_size, dtype=torch.long, device=device)
    #         step_beliefs = torch.zeros(batch_size, self.processes[0].num_states, dtype=torch.float32, device=device)

    #         for p_idx, process in enumerate(self.processes):
    #             mask = (active_process_indices == p_idx)
    #             if not mask.any():
    #                 continue
                
    #             # Extract states for this group
    #             group_states = current_states[mask]
    #             group_belief = current_belief[mask]
    #             group_size = group_states.shape[0]
                
    #             T = process._gpu_transition_matrix
                
    #             # --- Generation (Ground Truth) ---
    #             trans_probs = T[:, group_states, :].permute(1, 0, 2)
    #             flat_probs = trans_probs.reshape(group_size, -1)
    #             joint_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
                
    #             emission = joint_idx // process.num_states
    #             next_state = joint_idx % process.num_states
                
    #             # --- Filtering (Belief Update) ---
    #             T_emit = T[emission] # (group_size, num_states, num_states)
                
    #             # Belief Update: b_{t} = (b_{t-1} @ T_emit) / Normalization
    #             next_belief = torch.bmm(group_belief.unsqueeze(1), T_emit).squeeze(1)
    #             denom = next_belief.sum(dim=1, keepdim=True)
    #             group_belief = next_belief / torch.where(denom < 1e-12, torch.ones_like(denom), denom)

    #             # --- Store back ---
    #             # Apply vocab map immediately
    #             if process._gpu_vocab_map is not None:
    #                 final_emission = process._gpu_vocab_map[emission]
    #             else:
    #                 final_emission = emission

    #             step_emissions[mask] = final_emission
    #             step_true_states[mask] = next_state
    #             step_beliefs[mask] = group_belief
                
    #             current_states[mask] = next_state
    #             current_belief[mask] = group_belief
            
    #         emissions[:, t] = step_emissions
    #         true_states[:, t] = step_true_states
    #         beliefs[:, t] = step_beliefs

    #     return emissions, true_states, beliefs

    def derive_exact_mixed_tree_gpu(
        self,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
        max_beam_width: int = 100000,
        min_log_prob: float = -40.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Explores valid sequences on the GPU using Beam Search / Dynamic Pruning.
        This prevents O(V^L) memory explosion for long sequences (e.g., L=100) 
        by discarding impossible or astronomically unlikely paths at each step.
        
        Returns:
            sequences: (B_final, seq_len) Surviving discrete observation paths.
            seq_probs: (B_final,) The exact ground-truth probability of each surviving sequence.
            beliefs: (B_final, seq_len, num_states) The observer's marginal belief over states.
            active_procs: (B_final, seq_len, num_processes) The exact probability of which process is active.
        """
        # 1. Initialize Tensors
        for p in self.processes:
            p._ensure_gpu_tensors(device)

        vocab_len = self.processes[0].vocab_len
        num_procs = self.num_processes
        num_states = self.processes[0].num_states 

        T_emit_list = [p._gpu_transition_matrix for p in self.processes]
        T_next_list = []
        for p in self.processes:
            if hasattr(p, '_gpu_norm_transition_matrix') and p._gpu_norm_transition_matrix is not None:
                T_next_list.append(p._gpu_norm_transition_matrix)
            else:
                T_next_list.append(p._gpu_transition_matrix)

        # Batch starts at 1 (the empty sequence)
        B = 1
        sequences = torch.empty((B, 0), dtype=torch.long, device=device)
        
        # Use LOG probabilities to prevent float32 underflow over 100 steps
        seq_log_probs = torch.zeros(B, dtype=torch.float32, device=device) # log(1.0) = 0.0

        # Track active process probabilities (B, K)
        active_proc_probs = torch.zeros(B, num_procs, device=device)
        active_proc_probs[:, 0] = 1.0

        # Track beliefs per process (B, K, S)
        proc_beliefs = torch.zeros(B, num_procs, num_states, device=device)
        for k, p in enumerate(self.processes):
            if k == 0 and start_state_idx is not None:
                proc_beliefs[:, k, start_state_idx] = 1.0
            else:
                steady = p._get_gpu_steady_state(device)
                proc_beliefs[:, k, :] = steady

        # Outputs (will hold historical tensors)
        all_step_beliefs = []
        all_step_active_procs = []

        # Handle initial switch at t=0 before any observations
        if 0 in self.switch_schedule:
            p_switch = self.switch_schedule[0]
            new_active = torch.zeros_like(active_proc_probs)
            for k in range(num_procs):
                prev_k = (k - 1) % num_procs
                new_active[:, k] = (1 - p_switch) * active_proc_probs[:, k] + p_switch * active_proc_probs[:, prev_k]
            active_proc_probs = new_active

        # 2. Pruned Tensor Expansion Loop
        for t in range(seq_len):
            # A. Expand everything by V to test all next possible tokens
            sequences_next = sequences.repeat_interleave(vocab_len, dim=0)          
            seq_log_probs_next = seq_log_probs.repeat_interleave(vocab_len, dim=0)          
            active_proc_probs_next = active_proc_probs.repeat_interleave(vocab_len, dim=0) 
            proc_beliefs_next = proc_beliefs.repeat_interleave(vocab_len, dim=0)    

            B_expanded = sequences_next.shape[0]

            # B. Append the new combinatorial emissions
            emissions = torch.arange(vocab_len, device=device).repeat(B)
            sequences_next = torch.cat([sequences_next, emissions.unsqueeze(1)], dim=1)  

            # C. Observation Update (Filter)
            posterior_proc_probs = torch.zeros_like(active_proc_probs_next)
            evolved_beliefs = torch.zeros_like(proc_beliefs_next)
            p_emission = torch.zeros(B_expanded, device=device)

            for k in range(num_procs):
                T_emit_marginal = T_emit_list[k].sum(dim=2)           
                emission_likelihoods = T_emit_marginal[emissions]     

                # Likelihood: P(emission | process k)
                p_x_given_k = (proc_beliefs_next[:, k, :] * emission_likelihoods).sum(dim=1) 
                posterior_proc_probs[:, k] = p_x_given_k * active_proc_probs_next[:, k]

                # Evolve belief for process k
                T_sel = T_next_list[k][emissions]                     
                next_b = torch.einsum("bs, bsd -> bd", proc_beliefs_next[:, k, :], T_sel)
                
                # Normalize evolving belief
                evolved_beliefs[:, k, :] = next_b / (next_b.sum(dim=1, keepdim=True) + 1e-12)
                p_emission += posterior_proc_probs[:, k]

            # Update overall sequence LOG probability
            seq_log_probs_next = seq_log_probs_next + torch.log(p_emission + 1e-30)
            
            # Normalize active process probabilities
            active_proc_probs_next = posterior_proc_probs / (p_emission.unsqueeze(1) + 1e-12)

            # D. Mixed Process State Logic
            if self.state_mode == 'same':
                avg_belief = torch.zeros_like(proc_beliefs_next[:, 0, :])
                for k in range(num_procs):
                    w = active_proc_probs_next[:, k].unsqueeze(1)
                    avg_belief += evolved_beliefs[:, k, :] * w
                avg_belief = avg_belief / (avg_belief.sum(dim=1, keepdim=True) + 1e-12)
                for k in range(num_procs):
                    proc_beliefs_next[:, k, :] = avg_belief
            elif self.state_mode == 'resume':
                for k in range(num_procs):
                    w = active_proc_probs_next[:, k].unsqueeze(1)
                    proc_beliefs_next[:, k, :] = w * evolved_beliefs[:, k, :] + (1 - w) * proc_beliefs_next[:, k, :]
            elif self.state_mode == 'steady':
                for k in range(num_procs):
                    proc_beliefs_next[:, k, :] = evolved_beliefs[:, k, :]

            # ==========================================================
            # E. DYNAMIC PRUNING (The Beam Search Magic)
            # ==========================================================
            # 1. Mask out mathematically impossible (p_emission == 0) and highly unlikely paths
            valid_mask = (p_emission > 0.0) & (seq_log_probs_next > min_log_prob)
            
            # 2. Hard cutoff if we exceed GPU memory limits (Beam Width)
            if valid_mask.sum() > max_beam_width:
                top_k_thresh = torch.topk(seq_log_probs_next[valid_mask], max_beam_width).values[-1]
                valid_mask = valid_mask & (seq_log_probs_next >= top_k_thresh)

            # 3. Apply the mask to all current-step tracking tensors
            sequences = sequences_next[valid_mask]
            seq_log_probs = seq_log_probs_next[valid_mask]
            active_proc_probs = active_proc_probs_next[valid_mask]
            proc_beliefs = proc_beliefs_next[valid_mask]
            
            # 4. Apply the mask to historical step tensors to keep paths aligned
            for i in range(len(all_step_beliefs)):
                all_step_beliefs[i] = all_step_beliefs[i].repeat_interleave(vocab_len, dim=0)[valid_mask]
                all_step_active_procs[i] = all_step_active_procs[i].repeat_interleave(vocab_len, dim=0)[valid_mask]

            B_surviving = sequences.shape[0]

            # ==========================================================
            # F. Record Beliefs and Active Processes for the CURRENT step
            # ==========================================================
            marginal_belief = torch.zeros(B_surviving, num_states, device=device)
            for k in range(num_procs):
                marginal_belief += active_proc_probs[:, k].unsqueeze(1) * proc_beliefs[:, k, :]
            
            all_step_beliefs.append(marginal_belief)
            all_step_active_procs.append(active_proc_probs.clone())

            # ==========================================================
            # G. Switching Logic (Prior preparation for the NEXT step)
            # ==========================================================
            next_pos = t + 1
            if next_pos in self.switch_schedule:
                p_switch = self.switch_schedule[next_pos]
                new_active_probs = torch.zeros_like(active_proc_probs)
                new_beliefs = torch.zeros_like(proc_beliefs)

                for k in range(num_procs):
                    prev_k = (k - 1) % num_procs
                    prob_stay = (1 - p_switch) * active_proc_probs[:, k]
                    prob_arrive = p_switch * active_proc_probs[:, prev_k]
                    total_prob = prob_stay + prob_arrive
                    new_active_probs[:, k] = total_prob

                    if self.state_mode == 'steady':
                        steady_vec = self.processes[k]._get_gpu_steady_state(device).unsqueeze(0).expand(B_surviving, -1)
                        w_stay = (prob_stay / (total_prob + 1e-12)).unsqueeze(1)
                        w_arrive = (prob_arrive / (total_prob + 1e-12)).unsqueeze(1)
                        new_beliefs[:, k, :] = w_stay * proc_beliefs[:, k, :] + w_arrive * steady_vec
                    else:
                        new_beliefs[:, k, :] = proc_beliefs[:, k, :]

                active_proc_probs = new_active_probs
                proc_beliefs = new_beliefs
            
            # Optional: Print tracking progress if dealing with massive sequences
            # print(f"Step {t+1}: Tracking {B_surviving} valid sequences...")

            B = B_surviving

        # 3. Final Formatting
        # Convert log probabilities back to real probabilities for the final output
        seq_probs = torch.exp(seq_log_probs).float()
        
        all_step_beliefs = torch.stack(all_step_beliefs, dim=1)           # (B_final, L, S)
        all_step_active_procs = torch.stack(all_step_active_procs, dim=1) # (B_final, L, K)

        # Apply Vocab Mapping (e.g. converting 0,1,2 back to internal IDs if needed)
        if self.vocab_map is not None and self.processes[0]._gpu_vocab_map is not None:
            sequences = self.processes[0]._gpu_vocab_map[sequences]

        return sequences, seq_probs, all_step_beliefs, all_step_active_procs

class NormTransitionMixin:
    # GPU tensors for the norm matrix
    _gpu_norm_transition_matrix: Optional[torch.Tensor] = None
    def __init__(self):
        # Call Process.__init__()
        super().__init__()
        # Add extra matrix needed only in this subclass
        self.norm_transition_matrix = self._create_norm_matrix()
        if (
            len(self.norm_transition_matrix.shape) != 3
            or self.norm_transition_matrix.shape[1] != self.norm_transition_matrix.shape[2]
        ):
            raise ValueError(
                "Transition matrix should have 3 axes and the final two dims shoulds be square"
            )

        if self.norm_transition_matrix.shape[1] != self.norm_transition_matrix.shape[2]:
            raise ValueError("Transition matrix should be square")

        transition = self.norm_transition_matrix.sum(axis=0)
        # if not np.allclose(transition.sum(axis=1), 1.0):
        #     raise ValueError("Transition matrix should be stochastic and sum to 1")
        # Reset GPU cache for norm matrix
        self._gpu_norm_transition_matrix = None
        

    @abstractmethod
    def _create_norm_matrix(
        self,
    ) -> Float[np.ndarray, "vocab_len num_states num_states"] :
        """
        Create the HMM which defines the process.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices.
        """
        ...
    def _ensure_gpu_tensors(self, device: torch.device):
        """Override to also initialize norm transition matrix on GPU."""
        # Call parent's _ensure_gpu_tensors
        super()._ensure_gpu_tensors(device)
        # Also initialize norm transition matrix
        if self._gpu_norm_transition_matrix is None or self._gpu_device != device:
            self._gpu_norm_transition_matrix = torch.tensor(
                self.norm_transition_matrix, dtype=torch.float32, device=device
            )
    def generate_batch_gpu(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        start_state_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a batch of sequences on GPU for NormTransitionMixin processes.
        
        For these processes:
        - Emission is sampled from transition_matrix (marginal over next states)
        - Next state is sampled from norm_transition_matrix given the emission
        """
        self._ensure_gpu_tensors(device)
        
        T = self._gpu_transition_matrix  # (vocab_len, num_states, num_states)
        T_norm = self._gpu_norm_transition_matrix  # (vocab_len, num_states, num_states)
        
        # Sample initial states for all sequences: (batch_size,)
        if start_state_idx is not None:
            assert 0 <= start_state_idx < self.num_states, f"Invalid start_state_idx: {start_state_idx}"
            current_states = torch.full(
                (batch_size,), 
                start_state_idx, 
                dtype=torch.long, 
                device=device
            )
        else:
            steady_state = self._get_gpu_steady_state(device)  # (num_states,)
            current_states = torch.multinomial(
                steady_state.unsqueeze(0).expand(batch_size, -1), 
                num_samples=1
            ).squeeze(-1)  # (batch_size,)
        
        
        
        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        
        for t in range(seq_len):
            # Get transition probs for current states
            # T[:, current_states, :] -> (vocab_len, batch_size, num_states)
            trans_probs = T[:, current_states, :]  # (vocab_len, batch_size, num_states)
            
            # Marginalize over next states to get emission probs: (batch_size, vocab_len)
            emission_probs = trans_probs.sum(dim=2).permute(1, 0)  # (batch_size, vocab_len)
            
            # Sample emissions
            emission = torch.multinomial(emission_probs, num_samples=1).squeeze(-1)  # (batch_size,)
            if self._gpu_vocab_map is not None:
                emission[:,t] = self._gpu_vocab_map[emission]
            else:
                emission[:,t] = emission    
            
            # Now sample next state from norm_transition_matrix given emission
            # T_norm[emission, current_states, :] -> need to gather properly
            # For each sample i, we need T_norm[emission[i], current_states[i], :]
            
            # Efficient gather: (vocab_len, num_states, num_states) -> index by emission and current_states
            # T_norm[emission] gives (batch_size, num_states, num_states)
            next_state_probs = T_norm[emission, current_states, :]  # (batch_size, num_states)
            
            # Sample next states
            next_state = torch.multinomial(next_state_probs, num_samples=1).squeeze(-1)  # (batch_size,)
            current_states = next_state
        
        return emissions

    def _sample_emission_and_next_state(self, current_state_idx: int):
        """Override the Process version with the linear-process version."""
        transition_probs = self.transition_matrix[:, current_state_idx, :]

        emission_next_state_idx = np.random.choice(
            transition_probs.size, p=transition_probs.ravel()
        )

        emission = emission_next_state_idx // self.num_states


        next_state_idx = np.random.choice(
            self.num_states,
            p=self.norm_transition_matrix[emission, current_state_idx, :]
        )

        return emission, next_state_idx
    
    def derive_mixed_state_presentation(self, depth: int, start_state_idx: Optional[int] = None) -> MixedStateTree:
        if start_state_idx is not None:
            assert 0 <= start_state_idx < self.num_states, "start_state_idx out of bounds"
            initial_dist = np.zeros(self.num_states)
            initial_dist[start_state_idx] = 1.0
        else:
            initial_dist = self.steady_state_vector
        tree_root = MixedStateTreeNode(
            state_prob_vector= initial_dist,
            children=set(),
            path=[],
            emission_prob=0,
        )
        nodes = set([tree_root])

        stack: deque[
            tuple[MixedStateTreeNode, Float[np.ndarray, "num_states"], list[int], int]
        ] = deque([(tree_root, initial_dist, [], 0)])
        while stack:
            current_node, state_prob_vector, current_path, current_depth = stack.pop()
            if current_depth < depth:
                emission_probs = self._compute_emission_probabilities(
                    self, state_prob_vector
                )
                for emission in range(self.vocab_len):
                    if emission_probs[emission] > 0:
                        next_state_prob_vector = self._compute_next_distribution(
                            self.norm_transition_matrix, state_prob_vector, emission
                        )
                        child_path = current_path + [emission]
                        child_node = MixedStateTreeNode(
                            state_prob_vector=next_state_prob_vector,
                            path=child_path,
                            children=set(),
                            emission_prob=emission_probs[emission],
                        )
                        current_node.add_child(child_node)

                        stack.append(
                            (
                                child_node,
                                next_state_prob_vector,
                                child_path,
                                current_depth + 1,
                            )
                        )
            nodes.add(current_node)

        return MixedStateTree(
            root_node=tree_root, process=self.name, nodes=nodes, depth=depth
        )
    
    def _compute_emission_probabilities(self,
    hmm: Process, state_prob_vector: Float[np.ndarray, "num_states"]
) -> Float[np.ndarray, "vocab_len"]:
        """
        Compute the probabilities associated with each emission given the current mixed state.
        """
        T = hmm.transition_matrix
        emission_probs = np.einsum("s,esd->ed", state_prob_vector, T).sum(axis=1)
        emission_probs /= emission_probs.sum()
        return emission_probs


    def _compute_next_distribution(self,
        epsilon_machine: Float[np.ndarray, "vocab_len num_states num_states"],
        current_state_prob_vector: Float[np.ndarray, "num_states"],
        current_emission: int,
    ) -> Float[np.ndarray, "num_states"]:
            """
            Compute the next mixed state distribution for a given output.
            """
            X_next = np.einsum(
                "sd, s -> d", epsilon_machine[current_emission], current_state_prob_vector
            )
            return X_next 



