import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
import numpy as np

def get_kernel(X):
    K = []
    for gamma in [0.01, 0.05, 0.1, 1, 10, 50, 100]:
        K_rbf = rbf_kernel(X, gamma=gamma)
        K_rbf_std = StandardizeKernel(K_rbf, center=True, scale=True)
        K.append(K_rbf_std)

    poly_params = [(0, 2), (0, 4), (1, 2), (1, 4)]  
    for a, b in poly_params:
        K_poly = polynomial_kernel(X, degree=b, coef0=a, gamma=1)
        K_poly_std = StandardizeKernel(K_poly, center=True, scale=True)
        K.append(K_poly_std)

    K_linear = linear_kernel(X)  
    K_linear_std = StandardizeKernel(K_linear, center=True, scale=True)
    K.append(K_linear_std)
    K_tensor = np.stack(K, axis=2)
    return K_tensor

def StandardizeKernel(x, center = True, scale = True):

    assert isinstance(center, bool), "center must be boolean"
    assert isinstance(scale, bool), "scale must be boolean"
    assert not np.isnan(x).any(), "NA/NaN in argument."
    assert np.issubdtype(x.dtype, np.number), "argument is not numeric."
    assert np.isfinite(x).all(), "Inf/-Inf in argument."
    assert np.allclose(x, x.T), "x is not symmetric"
    eigvals = np.linalg.eigvalsh(x) 
    assert np.all(eigvals > -1e-9), "argument is not positive semi-definite."

    res = np.array(x)

    res_line = res.shape[0]
    assert res_line > 1, "Kernel matrix must have more than 2 rows"

    if center:
        J = np.ones((res_line,res_line)) / res_line
        res = res - J @ res - res @ J + J @ res @ J
    if scale:
        res = res / np.trace(res)

    return res 

def WithinClusterSS(K, H):
    """Calculate within-cluster sum of squares"""
    return np.trace(K) - np.trace(H.T @ K @ H)

def BtwClusterSS(K, H):
    """Calculate between-cluster sum of squares"""
    return np.trace(H.T @ K @ H)

def LabelToBinaryMat(x):
    """
    Convert cluster labels to binary indicator matrix
    x: 1D array of cluster labels
    return: n x c binary matrix
    """
    x = np.asarray(x).astype(str)
    n = x.shape[0]
    labels, counts = np.unique(x, return_counts=True)
    K = len(labels)
    res = np.zeros((n, K))
    for lb, label in enumerate(labels):
        res[x == label, lb] = 1 / np.sqrt(counts[lb])
    return res

def mkc(K, c, iter=10, epsilon = 1e-04, theta = None):
    assert c > 0 and int(c) == c, "Number of clusters must be positive integer"
    assert isinstance(K, np.ndarray) and K.ndim == 3, "K must be N x N x P 3D array"
    N, _, P = K.shape
    assert P >= 1, "K must have at least one view"

    # Initialize kernel weights theta
    if theta is None:
        theta = np.full(P, 1.0 / P)
    
    theta0 = theta.copy()
    Ktheta = np.zeros((N,N))
    ## Initialize combined kernel matrix
    for m in range(P):
        assert not np.isnan(K[:,:,m]).any(), f"NaN exists in kernel matrix of view {m+1}"
        Ktheta += K[:,:,m] * theta[m]
    
    ## Initialize indicator matrix H
    eigvals, eigvecs = np.linalg.eigh(Ktheta)
    id = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:,id]
    eigvals = eigvals[id]
    H = eigvecs[:,:c]

    ## Iterative optimization
    for it in range(iter):
        if(it == 0):
            theta0 = theta.copy()
        ## Update kernel weights theta
        else:
            theta0 = theta.copy()
            new_theta = []
            vals = []
            for m in range(P):
                vals.append(WithinClusterSS(K[:,:,m], H))
                new_theta.append(WithinClusterSS(K[:,:,m], H))  ## Within-cluster sum of squares in different views
            theta = np.array(new_theta)
            theta = theta / np.sqrt(np.sum(theta**2))
        
        ## Update combined kernel matrix
        Ktheta = np.zeros((N,N))
        for m in range(P):
            Ktheta += theta[m] * K[:, :, m]

        ## Update indicator matrix H
        eigvals, eigvecs = np.linalg.eigh(Ktheta)
        id = np.argsort(eigvals)[::-1]
        eigvals = eigvals[id]
        eigvecs = eigvecs[:,id]
        H = eigvecs[:,:c]

        ## Check convergence
        if(np.linalg.norm(theta0 - theta) < epsilon and it > 1):
            break
        if it == iter - 1:
            print(f"No convergence after {it+1} iterations")

    state = {}
    
    state['iter'] = it + 1
    state['H'] = H
    state['eigvecs'] = eigvecs
    state['eigvals'] = eigvals

    return state