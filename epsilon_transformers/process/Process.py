import numpy as np
from typing import Iterator, Optional
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
        assert len(self.symbols) == len(
            self.states
        ), "length of symbols & states must be the same"

    def __len__(self):
        return len(self.states)


class Process(ABC):
    name: str
    transition_matrix: Float[np.ndarray, "vocab_len num_states num_states"]
    state_names_dict: dict[str, int]
    vocab_len: int
    num_states: int
    # GPU tensors (lazily initialized)
    _gpu_transition_matrix: Optional[torch.Tensor] = None
    _gpu_steady_state: Optional[torch.Tensor] = None
    _gpu_device: Optional[torch.device] = None

    @property
    def steady_state_vector(self) -> Float[np.ndarray, "num_states"]:
        state_transition_matrix = np.sum(self.transition_matrix, axis=0)

        eigenvalues, eigenvectors = np.linalg.eig(state_transition_matrix.T)
        steady_state_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        normalized_steady_state_vector = steady_state_vector / steady_state_vector.sum()
        out: np.ndarray = normalized_steady_state_vector[:, 0]

        assert out.ndim == 1
        assert len(out) == self.num_states
        return out

    @property
    def is_unifilar(self) -> bool:
        # For each state, check if there are multiple transitions for each symbol
        for i in range(self.num_states):
            for j in range(self.vocab_len):
                # If there are multiple transitions, return False
                if np.count_nonzero(self.transition_matrix[j, i, :]) > 1:
                    return False
        return True

    def __init__(self):
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
        # Reset GPU cache
        self._gpu_transition_matrix = None
        self._gpu_steady_state = None
        self._gpu_device = None

    @abstractmethod
    def _create_hmm(
        self,
    ) -> tuple[Float[np.ndarray, "vocab_len num_states num_states"], dict[str, int]]:
        """
        Create the HMM which defines the process.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices.
        """
        ...

    def __str__(self):
        return (
            f"{self.name} Process\n"
            f"Number of states: {self.num_states}\n"
            f"Vocabulary length: {self.vocab_len}\n"
            f"Transition matrix shape: {self.transition_matrix.shape}"
        )
    def _ensure_gpu_tensors(self, device: torch.device):
        """Lazily initialize GPU tensors for batch generation."""
        if self._gpu_device != device or self._gpu_transition_matrix is None:
            self._gpu_transition_matrix = torch.tensor(
                self.transition_matrix, dtype=torch.float32, device=device
            )
            self._gpu_steady_state = torch.tensor(
                self.steady_state_vector, dtype=torch.float32, device=device
            )
            self._gpu_device = device

    def generate_batch_gpu(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate a batch of sequences on GPU in parallel.
        
        Args:
            batch_size: Number of sequences to generate
            seq_len: Length of each sequence
            device: torch device (cuda, mps, or cpu)
            
        Returns:
            torch.Tensor of shape (batch_size, seq_len) with emissions (dtype=long)
        """
        self._ensure_gpu_tensors(device)
        
        T = self._gpu_transition_matrix  # (vocab_len, num_states, num_states)
        steady_state = self._gpu_steady_state  # (num_states,)
        
        # Sample initial states for all sequences: (batch_size,)
        current_states = torch.multinomial(
            steady_state.unsqueeze(0).expand(batch_size, -1), 
            num_samples=1
        ).squeeze(-1)  # (batch_size,)
        
        emissions = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
        
        for t in range(seq_len):
            # Get transition probs for current states: (batch_size, vocab_len, num_states)
            # T is (vocab_len, num_states, num_states)
            # We need T[:, current_states[i], :] for each i
            trans_probs = T[:, current_states, :]  # (vocab_len, batch_size, num_states)
            trans_probs = trans_probs.permute(1, 0, 2)  # (batch_size, vocab_len, num_states)
            
            # Flatten to (batch_size, vocab_len * num_states)
            flat_probs = trans_probs.reshape(batch_size, -1)
            
            # Sample emission and next state jointly
            joint_idx = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # (batch_size,)
            
            # Decode emission and next state
            emission = joint_idx // self.num_states
            next_state = joint_idx % self.num_states
            
            emissions[:, t] = emission
            current_states = next_state
        
        return emissions
       
    def _sample_emission(self, current_state_idx: int | None = None) -> int:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )

        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"

        p = self.transition_matrix[:, current_state_idx, :].sum(axis=1)
        emission = np.random.choice(self.vocab_len, p=p)
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
        transition_probs = self.transition_matrix[:, current_state_idx, :] # (vocab_len, state)
        emission_next_state_idx = np.random.choice(
            transition_probs.size, p=transition_probs.ravel()
        )
        emission = emission_next_state_idx // self.num_states
        next_state_idx = emission_next_state_idx % self.num_states
        return emission, next_state_idx

    def yield_emission_histories(
        self, sequence_len: int, num_sequences: int
    ) -> Iterator[list[int]]:
        for _ in range(num_sequences):
            yield [x for x in self.yield_emissions(sequence_len=sequence_len)]

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
    def derive_mixed_state_presentation(self, depth: int) -> MixedStateTree:
        tree_root = MixedStateTreeNode(
            state_prob_vector=self.steady_state_vector,
            children=set(),
            path=[],
            emission_prob=0,
        )
        nodes = set([tree_root])

        stack: deque[
            tuple[MixedStateTreeNode, Float[np.ndarray, "num_states"], list[int], int]
        ] = deque([(tree_root, self.steady_state_vector, [], 0)])
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
        steady_state = self._gpu_steady_state  # (num_states,)
        
        # Sample initial states for all sequences: (batch_size,)
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
            emissions[:, t] = emission
            
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
    
    def derive_mixed_state_presentation(self, depth: int) -> MixedStateTree:
        tree_root = MixedStateTreeNode(
            state_prob_vector=self.steady_state_vector,
            children=set(),
            path=[],
            emission_prob=0,
        )
        nodes = set([tree_root])

        stack: deque[
            tuple[MixedStateTreeNode, Float[np.ndarray, "num_states"], list[int], int]
        ] = deque([(tree_root, self.steady_state_vector, [], 0)])
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



