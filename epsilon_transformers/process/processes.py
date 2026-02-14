import numpy as np
from typing import Optional, cast
from types import FrameType
import inspect

from epsilon_transformers.process.Process import Process,NormTransitionMixin

# TODO: Write test for PROCESS_REGISTRY
# TODO: Think if you really need PROCESS_REGSITRY (if only getting called during dataloader creation, it may be better to have the dataloader take in a process)
# TODO: Add test to make sure that all members of this module are a member of Process
# TODO: Find paper where mess3 process is introduced
# TODO: Think through whether self.name is necessary (review it's usage in derive_mixed_state_presentation)
# TODO: Move _create_hmm into the init function prior to super()__init__(**kwargs)


class ZeroOneR(Process):
    def __init__(self, prob_of_zero_from_r_state: float = 0.5,**kwargs):
        self.name = "z1r"
        self.p = prob_of_zero_from_r_state
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((2, 3, 3))
        state_names = {"0": 0, "1": 1, "R": 2}
        T[0, state_names["0"], state_names["1"]] = 1.0
        T[1, state_names["1"], state_names["R"]] = 1.0
        T[0, state_names["R"], state_names["0"]] = self.p
        T[1, state_names["R"], state_names["0"]] = 1 - self.p

        return T, state_names


class RRXOR(Process):
    def __init__(self, pR1=0.5, pR2=0.5,**kwargs):
        self.name = "rrxor"
        self.pR1 = pR1
        self.pR2 = pR2
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((2, 5, 5))
        state_names = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
        T[0, state_names["S"], state_names["0"]] = self.pR1
        T[1, state_names["S"], state_names["1"]] = 1 - self.pR1
        T[0, state_names["0"], state_names["F"]] = self.pR2
        T[1, state_names["0"], state_names["T"]] = 1 - self.pR2
        T[0, state_names["1"], state_names["T"]] = self.pR2
        T[1, state_names["1"], state_names["F"]] = 1 - self.pR2
        T[1, state_names["T"], state_names["S"]] = 1.0
        T[0, state_names["F"], state_names["S"]] = 1.0

        return T, state_names

class NoisyRRXOR(Process):
    def __init__(self, pR1=0.5, pR2=0.5, epsilon=0.02,**kwargs):
        self.name = "noisy_rrxor"
        self.pR1 = pR1
        self.pR2 = pR2
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def _create_hmm(self):
        # 1. Create the base RRXOR Transition Matrix (2, 5, 5)
        # Using the logic from your RRXOR class
        T_base = np.zeros((2, 5, 5))
        state_names = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
        
        T_base[0, state_names["S"], state_names["0"]] = self.pR1
        T_base[1, state_names["S"], state_names["1"]] = 1 - self.pR1
        T_base[0, state_names["0"], state_names["F"]] = self.pR2
        T_base[1, state_names["0"], state_names["T"]] = 1 - self.pR2
        T_base[0, state_names["1"], state_names["T"]] = self.pR2
        T_base[1, state_names["1"], state_names["F"]] = 1 - self.pR2
        T_base[1, state_names["T"], state_names["S"]] = 1.0
        T_base[0, state_names["F"], state_names["S"]] = 1.0

        # 2. Apply the Noise Transformation
        # Formula: T_new = (1 - epsilon) * T + (epsilon / (2 * |S|)) * Ones_Matrix
        num_states = 5
        noise_term = (self.epsilon / (2 * num_states)) * np.ones((5, 5))
        
        # Apply to each symbol's transition matrix T[x]
        T_noisy = np.zeros_like(T_base)
        for x in range(2):
            T_noisy[x] = (1 - self.epsilon) * T_base[x] + noise_term
            
        return T_noisy, state_names
class Trun_Mess3(Process):
    def __init__(self, x=0.15, a=0.6,r=1,t1=1,t2=2,**kwargs):
        self.name = "trun_mess3"
        self.x = x
        self.a = a
        self.r = r
        self.t1 = t1
        self.t2 = t2
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x


        T[0, :, :] = [[0, 0, bx/2], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by+ay, ax+bx, 1.5*bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]


        return T,state_names   

class Mixed_Mess3(Process):
    def __init__(self, x1=0.15, a1=0.6,x2=0.15,a2=0.6,**kwargs):
        self.name = "mixed_mess3"
        self.x1 = x1
        self.a1 = a1
        self.x2 = x2
        self.a2 = a2
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((6, 6, 6))
        state_names = {"A": 0, "B": 1, "C": 2,"D": 3,"E": 4,"F": 5}
        b1 = (1 - self.a1) / 2
        y1 = 1 - 2 * self.x1
        b2 = (1 - self.a2) / 2
        y2 = 1 - 2 * self.x2

        ay1 = self.a1 * y1
        bx1 = b1 * self.x1
        by1 = b1 * y1
        ax1 = self.a1 * self.x1
        ay2 = self.a2 * y2
        bx2 = b2 * self.x2
        by2 = b2 * y2
        ax2 = self.a2 * self.x2

        T[0, :, :] = [[ay1, bx1, bx1,0,0,0], [ax1, by1, bx1,0,0,0], [ax1, bx1, by1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
        T[1, :, :] = [[by1, ax1, bx1,0,0,0], [bx1, ay1, bx1,0,0,0], [bx1, ax1, by1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
        T[2, :, :] = [[by1, bx1, ax1,0,0,0], [bx1, by1, ax1,0,0,0], [bx1, bx1, ay1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]] 
        T[3, :, :] = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,ay2, bx2, bx2], [0,0,0,ax2, by2, bx2], [0,0,0,ax2, bx2, by2]]
        T[4, :, :] = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,by2, ax2, bx2], [0,0,0,bx2, ay2, bx2], [0,0,0,bx2, ax2, by2]]
        T[5, :, :] = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,by2, bx2, ax2], [0,0,0,bx2, by2, ax2], [0,0,0,bx2, bx2, ay2]]
        return T,state_names


        

        return T,state_names
class Linear_Mess3(NormTransitionMixin,Process):
    def __init__(self, x=0.15, a=0.6,**kwargs):
        self.name = "linear_mess3"
        self.x = x
        self.a = a
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]


        return T,state_names
    
    def _create_norm_matrix(self):
        
        T_n = np.zeros((3,3,3))
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        ay2bx=ay+2*bx
        axbybx=ax+by+bx

        T_n[0, :, :] = [[ay/ay2bx, bx/ay2bx, bx/ay2bx], [ax/axbybx, by/axbybx, bx/axbybx], [ax/axbybx, bx/axbybx, by/axbybx]]
        T_n[1, :, :] = [[by/axbybx, ax/axbybx, bx/axbybx], [bx/ay2bx, ay/ay2bx, bx/ay2bx], [bx/axbybx, ax/axbybx, by/axbybx]]
        T_n[2, :, :] = [[by/axbybx, bx/axbybx, ax/axbybx], [bx/axbybx, by/axbybx, ax/axbybx], [bx/ay2bx, bx/ay2bx, ay/ay2bx]]

        return T_n
    
class Mess3(Process):
    def __init__(self, x=0.15, a=0.6,**kwargs):
        self.name = "mess3"
        self.x = x
        self.a = a
        super().__init__(**kwargs)

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]

        return T, state_names    
    


class Even(Process):
    def __init__(self,**kwargs):
        self.name = "Even"
        super().__init__(**kwargs)

    def _create_hmm(self):
        state_names = {"0": 0, "1": 1}
        T = np.zeros((2, 2, 2))
        T[0,0,0] = 0.5   # From state 0, emit 0, go to state 0
        T[1,0,1] = 0.5   # From state 0, emit 1, go to state 1
        T[1,1,0] = 1.0   # From state 1, emit 1, go to state 0
        return T, state_names

class GoldenMean(Process):
    def __init__(self,**kwargs):
        self.name = "Golden"
        super().__init__(**kwargs)

    def _create_hmm(self):
        state_names = {"0": 0, "1": 1}
        T = np.zeros((2, 2, 2))
        T[0,0,0] = 0.5  # From state 0, emit 0, go to state 0
        T[1,0,1] = 0.5  # From state 0, emit 1, go to state 1
        T[0,1,0] = 1.0  # From state 1, emit 0, go to state 0 
        return T, state_names
 
class TransitionMatrixProcess(Process):
    def __init__(self, transition_matrix: np.ndarray,**kwargs):
        self.transition_matrix = transition_matrix
        super().__init__(**kwargs)

    def _create_hmm(self):
        return self.transition_matrix, {
            i: i for i in range(self.transition_matrix.shape[0])
        }


class SRA(Process):
    """
    Subject-Relation-Attribute Process (SRA).
    Generates triplets of (Subject, Relation, Attribute) consistently.
    
    Vocabulary Organization:
    - 0 to R-1: Relation tokens
    - R + i*(R+1): Subject i token
    - R + i*(R+1) + 1 + j: Attribute token for Subject i and Relation j
    
    Topology:
    - ROOT: Ready to emit Subject.
    - HOLD_S[i]: Holding Subject i. Ready to emit Relation.
    - HOLD_SR[i][j]: Holding Subject i and Relation j. Ready to emit Attribute.
    """
    def __init__(self, num_subjects=10, num_relations=2,relation_probs: list[float]|None=None,**kwargs):
        self.name = "sra"
        self.S = num_subjects
        self.R = num_relations
        if relation_probs is None:
            self.relation_probs=[1.0/self.R]*self.R
        else :
            if len(relation_probs) != self.R:
                    raise ValueError(f"Length of relation_probs ({len(relation_probs)}) must match num_relations ({self.R})")
            if not np.isclose(sum(relation_probs), 1.0):
                    # Optional: normalize automatically or raise error. Raising error is safer for research.
                    raise ValueError(f"relation_probs must sum to 1.0, got {sum(relation_probs)}")
            self.relation_probs = relation_probs       
        super().__init__(**kwargs)


    def _create_hmm(self):
        # Vocab Size = R (relations) + S (subjects) + S*R (attributes)
        # Actually structure is S blocks of (1 Subject + R Attributes)
        vocab_size = self.R + self.S * (1 + self.R)
        
        # States: 
        # 0: ROOT (Expect Subject)
        # 1..S: Expect Relation (After Subject i)
        # S+1..S+S*R: Expect Attribute (After Subject i, Relation j)
        num_states = 1 + self.S + (self.S * self.R)
        
        T = np.zeros((vocab_size, num_states, num_states))
        state_names = {"ROOT": 0}
        
        # --- Helper for Indices ---
        def get_subj_state_idx(s_idx):
            return 1 + s_idx
            
        def get_subj_rel_state_idx(s_idx, r_idx):
            return 1 + self.S + (s_idx * self.R) + r_idx

        def get_subj_token(s_idx):
            return self.R + s_idx * (1 + self.R)
            
        def get_attr_token(s_idx, r_idx):
            return self.R + s_idx * (1 + self.R) + 1 + r_idx

        # --- 1. From ROOT, Emit Subject i -> Go to HOLD_S[i] ---
        # Uniform probability over all subjects
        prob_s = 1.0 / self.S
        for s in range(self.S):
            token = get_subj_token(s)
            next_state = get_subj_state_idx(s)
            T[token, 0, next_state] = prob_s
            state_names[f"Hold_S{s}"] = next_state

        # --- 2. From HOLD_S[i], Emit Relation j -> Go to HOLD_SR[i][j] ---
        # Uniform probability over all relations
        # prob_r = 1.0 / self.R
        for s in range(self.S):
            current_state = get_subj_state_idx(s)
            for r in range(self.R):
                token = r # Relation tokens are 0..R-1
                next_state = get_subj_rel_state_idx(s, r)
                T[token, current_state, next_state] = self.relation_probs[r]
                state_names[f"Hold_S{s}_R{r}"] = next_state

        # --- 3. From HOLD_SR[i][j], Emit Attribute(i, j) -> Go to ROOT ---
        # Deterministic emission of the correct attribute
        for s in range(self.S):
            for r in range(self.R):
                current_state = get_subj_rel_state_idx(s, r)
                token = get_attr_token(s, r)
                T[token, current_state, 0] = 1.0

        return T, state_names

    
PROCESS_REGISTRY: dict[str, type] = {
    key: value
    # cast because we know the current frame has the above classes
    for key, value in cast(FrameType, inspect.currentframe()).f_locals.items()
    if isinstance(value, type) and issubclass(value, Process) and key != "Process"
}
