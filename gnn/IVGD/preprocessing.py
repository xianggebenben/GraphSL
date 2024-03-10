from typing import List, Tuple, Dict
import copy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def gen_seeds(size: int = None) -> np.ndarray:
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(
            max_uint32 + 1, size=size, dtype=np.uint32)


def exclude_idx(idx: np.ndarray, idx_exclude_list: List[np.ndarray]) -> np.ndarray:
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])


def gen_splits_(array, train_size, stopping_size, val_size):
    # import ipdb; ipdb.set_trace()
    assert train_size + stopping_size + val_size <= len(array), 'length error'
    from sklearn.model_selection import train_test_split
    train_idx, tmp = train_test_split(array, train_size=train_size, test_size=stopping_size+val_size)
    stopping_idx, val_idx = train_test_split(tmp, train_size=stopping_size, test_size=val_size)
    
    return train_idx, stopping_idx, val_idx

def normalize_attributes(attr_matrix):
    epsilon = 1e-12
    if isinstance(attr_matrix, sp.csr_matrix):
        attr_norms = spla.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])
    else:
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm
