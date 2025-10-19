import numpy as np
from typing import Dict
from .framework import ArgumentationFramework
from .hc_semantics import weighted_h_categorizer

def sample_weights(m: int, n_samples: int, seed: int = 0) -> np.ndarray:
    '''draw n sample random weiht vector'''
    rng = np.random.default_rng(int(seed))
    return rng.random((n_samples, m)) 

def transform_to_acceptability(
    af: ArgumentationFramework,
    W: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    '''Map a batch of weight vectors to acceptability vectors via HC'''
    n, m = W.shape
    X = np.zeros_like(W)
    A = af.A
    for i in range(n):
        # Build weight dict for the i-th sample: argument name -> weight
        w_dict: Dict[str, float] = {a: W[i, j] for j, a in enumerate(A)}
        x_dict = weighted_h_categorizer(af, w_dict, epsilon=epsilon)
        # Store results in the same argument order as A
        X[i] = [x_dict[a] for a in A]
    return X
