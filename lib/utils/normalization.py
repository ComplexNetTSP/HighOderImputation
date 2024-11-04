import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy import sparse


def easy_inverse_matrix(D):
    diagonal_elements = np.diag(D)
    diagonal_elements_inv = 1/diagonal_elements
    diagonal_elements_inv[np.isinf(diagonal_elements_inv) | np.isneginf(diagonal_elements_inv)] = 0
    D_inv = np.diag(diagonal_elements_inv, 0)
    return D_inv



def compute_bunch_normalized_matrices(
    B1: np.ndarray | csr_matrix, B2: np.ndarray | csr_matrix
) -> tuple[
    np.ndarray | csr_matrix,
    np.ndarray | csr_matrix,
    np.ndarray | csr_matrix,
    np.ndarray | csr_matrix,
]:
    """Get Bunch normalization.

    Parameters
    ----------
    B1 : np.ndarray or csr_matrix
        The boundary B1: C1->C0 of a complex.
    B2 : np.ndarray or csr_matrix
        The boundary B2: C2->C1 of a complex.

    Returns
    -------
    B1 : np.ndarray or csr_matrix
        Normalized B1: C1->C0.
    B1T : np.ndarray or csr_matrix
        Normalized B1T: C0->C1.
    B2 : np.ndarray or csr_matrix
        Normalized B2: C2->C1.
    B2T : np.ndarray or csr_matrix
        Normalized B2T: C1->C2.

    References
    ----------
    .. [1] Schaub, Benson, Horn, Lippner, Jadbabaie.
        Random walks on simplicial complexes and the normalized hodge 1-laplacian.
        Paper: https://epubs.siam.org/doi/10.1137/18M1201019

    .. [2] Bunch, You, Fung, Singh.
        Simplicial 2-Complex Convolutional Neural Networks.
        Paper: https://openreview.net/forum?id=TLbnsKrt6J-
    """
    B1N = _compute_B1_normalized_matrix(B1, B2)
    B1TN = _compute_B1T_normalized_matrix(B1, B2)
    B2N = _compute_B2_normalized_matrix(B2)
    B2TN = _compute_B2T_normalized_matrix(B2)
    return B1N, B1TN, B2N, B2TN


def _compute_B1_normalized_matrix(
    B1: np.ndarray | csr_matrix, B2: np.ndarray | csr_matrix
) -> np.ndarray | csr_matrix:
    """Compute normalized boundary matrix B1.

    Parameters
    ----------
    B1 : np.ndarray or csr_matrix
        The boundary B1: C1->C0 of a complex.
    B2 : np.ndarray or csr_matrix
        The boundary B2: C2->C1 of a complex.

    Returns
    -------
    np.ndarray or csr_matrix
        Normalized B1: C1->C0.
    """
    D2 = _compute_D2(B2)
    D1 = _compute_D1(B1, D2)
    if isinstance(B1, np.ndarray):
        #D1_pinv = pinv(D1)
        D1_inv = easy_inverse_matrix(D1)
    elif isinstance(B1, csr_matrix):
        #D1_pinv = csr_matrix(pinv(D1.toarray()))
        D1_inv = csr_matrix(easy_inverse_matrix(D1.toarray()))

    return D1_inv @ B1
    #return D1_pinv @ B1


def _compute_B1T_normalized_matrix(
    B1: np.ndarray | csr_matrix, B2: np.ndarray | csr_matrix
) -> np.ndarray | csr_matrix:
    """Compute normalized transpose boundary matrix B1T.

    Parameters
    ----------
    B1 : np.ndarray or csr_matrix
        The boundary B1: C1->C0  of a complex.
    B2 : np.ndarray or csr_matrix
        The boundary B2: C2->C1  of a complex.

    Returns
    -------
    np.ndarray or csr_matrix
        Normalized transpose boundary matrix B1T: C0->C1.
        This is the coboundary C0->C1.
    """
    D2 = _compute_D2(B2)
    D1 = _compute_D1(B1, D2)
    if isinstance(B1, np.ndarray):
        #D1_pinv = pinv(D1)
        D1_inv = easy_inverse_matrix(D1)
    elif isinstance(B1, csr_matrix):
        #D1_pinv = csr_matrix(pinv(D1.toarray()))
        D1_inv = csr_matrix(easy_inverse_matrix(D1.toarray()))
    else:
        raise TypeError("input type must be either np.ndarray or csr_matrix")
    return D2 @ B1.T @ D1_inv
    #return D2 @ B1.T @ D1_pinv


def _compute_B2_normalized_matrix(
    B2: np.ndarray | csr_matrix,
) -> np.ndarray | csr_matrix:
    """Compute normalized boundary matrix B2.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary matrix B2: C2 -> C1 of a simplicial complex.

    Returns
    -------
    np.ndarray or csr_matrix
        Normalized boundary matrix B2: C2 -> C1.
    """
    D3 = _compute_D3(B2)
    return B2 @ D3


def _compute_B2T_normalized_matrix(
    B2: np.ndarray | csr_matrix,
) -> np.ndarray | csr_matrix:
    """Compute normalized transpose matrix operator B2T.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary B2: C2->C1.

    Returns
    -------
    np.ndarray or csr_matrix
        Normalized transpose matrix operator B2T: C1->C2.
        This is the coboundary matrix: C1->C2.
    """
    D5 = _compute_D5(B2)
    if isinstance(B2, np.ndarray):
        #D5_pinv = pinv(D5)
        D5_inv = easy_inverse_matrix(D5)
        #return B2.T @ D5_pinv
        return B2.T @ D5_inv
    if isinstance(B2, csr_matrix):
        #D5_pinv = csr_matrix(pinv(D5.toarray()))
        D5_inv = csr_matrix(easy_inverse_matrix(D5.toarray()))
        #return B2.T @ D5_pinv
        return B2.T @ D5_inv
    raise TypeError("input type must be either np.ndarray or csr_matrix")


def _compute_D1(B1: np.ndarray | csr_matrix, D2: diags) -> diags:
    """Compute the degree matrix D1.

    Parameters
    ----------
    B1 : np.ndarray or csr_matrix
        The boundary B1: C1->C0 of a complex.
    D2 : diags
        The degree matrix D2.

    Returns
    -------
    diags
        Normalized degree matrix D1.
    """
    if isinstance(B1, csr_matrix):
        rowsum = np.array((np.abs(B1) @ D2).sum(axis=1)).flatten()
        return 2 * diags(rowsum)
    if isinstance(B1, np.ndarray):
        rowsum = (np.abs(B1) @ D2).sum(axis=1)
        return 2 * np.diag(rowsum)
    raise TypeError("Input type must be either np.ndarray or csr_matrix.")


def _compute_D2(B2: np.ndarray | csr_matrix) -> diags:
    """Compute the degree matrix D2.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary matrix B2: C2 -> C1 of a simplicial complex.

    Returns
    -------
    diags
        Normalized degree matrix D2.
    """
    if isinstance(B2, csr_matrix):
        rowsum = np.array(np.abs(B2).sum(axis=1)).flatten()
        return diags(np.maximum(rowsum, 1))
    if isinstance(B2, np.ndarray):
        rowsum = np.abs(B2).sum(axis=1)
        return np.diag(np.maximum(rowsum, 1))
    raise TypeError("Input type must be either np.ndarray or csr_matrix.")


def _compute_D3(B2: np.ndarray | csr_matrix) -> diags:
    """Compute the degree matrix D3.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary matrix B2: C2 -> C1 of a simplicial complex.

    Returns
    -------
    diags
        Degree matrix D3.
    """
    if isinstance(B2, csr_matrix):
        return diags(np.ones(B2.shape[1]) / 3)
    if isinstance(B2, np.ndarray):
        return np.diag(np.ones(B2.shape[1]) / 3)
    raise TypeError("Input type must be either np.ndarray or csr_matrix.")


def _compute_D5(B2: np.ndarray | csr_matrix) -> diags:
    """Compute the degree matrix D5.

    Parameters
    ----------
    B2 : np.ndarray or csr_matrix
        The boundary matrix B2: C2 -> C1 of a simplicial complex.

    Returns
    -------
    diags
        Normalized degree matrix D5.
    """
    if isinstance(B2, csr_matrix):
        rowsum = np.array(np.abs(B2).sum(axis=1)).flatten()
        return diags(rowsum)
    if isinstance(B2, np.ndarray):
        rowsum = np.abs(B2).sum(axis=1)
        return np.diag(rowsum)
    raise TypeError("Input type must be either np.ndarray or csr_matrix.")


def normalized_high_order_adjcency(B1, B2):
    # L0
    L0 = B1 @ B1.T
    B1_sum = np.abs(B1).sum(axis=1)
    D0 = sparse.diags(B1_sum.reshape(-1), 0)
    B1_sum_inv = 1. / B1_sum
    B1_sum_inv[np.isinf(B1_sum_inv) | np.isneginf(B1_sum_inv)] = 0
    D0_inv = np.diag(B1_sum_inv.reshape(-1), 0)
    L0 = D0_inv @ L0
    L0factor = (-1) * np.diag((1 / (B1_sum_inv + 1)).reshape(-1), 0)
    L0bias = np.eye(D0.shape[0])
    L0 = L0factor @ L0 + L0bias

    # L1
    A_1u = _compute_D2(B2) - B2 @ _compute_D3(B2) @ B2.T

    # A_1d = pinv(_compute_D2(B2)) - B1.T@pinv(_compute_D1(B1, _compute_D2(B2)))@B1
    A_1d = easy_inverse_matrix(_compute_D2(B2)) - B1.T @ easy_inverse_matrix(_compute_D1(B1, _compute_D2(B2))) @ B1
    # easy_inverse_matrix

    A_1u_norm = (
                        A_1u
                        +
                        np.eye(A_1u.shape[0])
                ) @ np.diag(1 / _compute_D2(B2).diagonal() + 1, 0)

    A_1d_norm = (
                        _compute_D2(B2)
                        +
                        np.eye(_compute_D2(B2).shape[0])
                ) @ (A_1d + np.eye(A_1d.shape[0]))

    L1 = A_1u_norm + A_1d_norm
    # L2
    # A_2d = np.eye(B2.shape[1]) - B2.T@pinv(_compute_D5(B2))@B2
    A_2d = np.eye(B2.shape[1]) - B2.T @ easy_inverse_matrix(_compute_D5(B2)) @ B2

    L2 = (2 * np.eye(B2.shape[1])) @ (A_2d + np.eye(A_2d.shape[0]))

    return L0, L1, L2



