from typing import List
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def gen_seeds(size: int = None) -> np.ndarray:
    """
    Generate an array of random seeds.

    Args:

    - size (int, optional): Size of the array to generate. If None, a single random seed is returned.

    Returns:

    - np.ndarray: Array of random seeds.
    """
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(max_uint32 + 1, size=size, dtype=np.uint32)


def exclude_idx(idx: np.ndarray, idx_exclude_list: List[np.ndarray]) -> np.ndarray:
    """
    Exclude indices from a given array based on a list of arrays containing indices to exclude.

    Args:

    - idx (np.ndarray): Array of indices.

    - idx_exclude_list (List[np.ndarray]): List of arrays containing indices to exclude.

    Returns:

    - np.ndarray: Array of indices after exclusion.
    """
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])


def gen_splits_(array, train_size, stopping_size, val_size):
    """
    Generate train, stopping, and validation indices splits from a given array.

    Args:

    - array: Array of indices.

    - train_size (int): Size of the training split.

    - stopping_size (int): Size of the stopping split.

    - val_size (int): Size of the validation split.

    Returns:

    - train_idx (numpy.ndarray): Train indices splits.

    - stopping_idx (numpy.ndarray): Stopping indices splits.

    - val_idx (numpy.ndarray): Validation indices splits.

    """
    assert train_size + stopping_size + val_size <= len(array), 'length error'
    from sklearn.model_selection import train_test_split
    train_idx, tmp = train_test_split(
        array, train_size=train_size, test_size=stopping_size + val_size)
    stopping_idx, val_idx = train_test_split(
        tmp, train_size=stopping_size, test_size=val_size)

    return train_idx, stopping_idx, val_idx


def normalize_attributes(attr_matrix):
    """
    Normalize attributes in a matrix.

    Args:

    - attr_matrix: Matrix containing attributes to normalize.

    Returns:

    - attr_mat_norm (np.ndarray): Normalized attribute matrix.
    """
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
