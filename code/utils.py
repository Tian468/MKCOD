import numpy as np

def StandardizeKernel(x, center = True, scale = True):

    assert isinstance(center, bool), "center must be a bool"
    assert isinstance(scale, bool), "scale must be a bool"
    assert not np.isnan(x).any(), "NA/NaN in argument."
    assert np.issubdtype(x.dtype, np.number), "argument is not numeric."
    assert np.isfinite(x).all(), "Inf/-Inf in argument."
    assert np.allclose(x, x.T), "x must be symmetric"
    eigvals = np.linalg.eigvalsh(x) 
    if not np.all(eigvals > -1e-9):
        print("Warning: kernel matrix is numerically not PSD; continuing anyway.")

    res = np.array(x)

    res_line = res.shape[0]
    assert res_line > 1, "The kernel matrix must have more than one row"

    if center:
        J = np.ones((res_line,res_line)) / res_line
        res = res - J @ res - res @ J + J @ res @ J
    if scale:
        res = res / np.trace(res)

    return res 

def WithinClusterSS(K, H):
    """Calculate the within-cluster sum of squares."""
    return np.trace(K) - np.trace(H.T @ K @ H)


def LabelToBinaryMat(x):
    """
    Convert cluster labels to a binary indicator matrix.

    x: One-dimensional array of cluster labels.
    Returns: An n x k indicator matrix.
    """
    x = np.asarray(x).astype(str)
    n = x.shape[0]
    labels, counts = np.unique(x, return_counts=True)
    K = len(labels)
    res = np.zeros((n, K))
    for lb, label in enumerate(labels):
        res[x == label, lb] = 1 / np.sqrt(counts[lb])
    return res
