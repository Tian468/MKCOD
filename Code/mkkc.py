from utils import WithinClusterSS
import numpy as np
from sklearn.cluster import KMeans

def mkkcEst(K, c, iter=50, epsilon = 1e-04, theta = None):
    assert c > 0 and int(c) == c, "The number of clusters must be a positive integer"
    assert isinstance(K, np.ndarray) and K.ndim == 3, "K must be an N x N x P array"
    N, _, P = K.shape
    assert P >= 1, "K must contain at least one view"

    # Initialize the kernel weights.
    if theta is None:
        theta = np.full(P, 1.0 / P)
    
    theta0 = theta.copy()
    Ktheta = np.zeros((N,N))
    # Initialize the combined kernel matrix.
    for m in range(P):
        assert not np.isnan(K[:,:,m]).any(), f"Kernel view {m + 1} contains NaN values"
        Ktheta += K[:,:,m] * theta[m]
    
    # Initialize the indicator matrix H.
    eigvals, eigvecs = np.linalg.eigh(Ktheta)
    id = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:,id]
    eigvals = eigvals[id]
    H = eigvecs[:,:c]

    # Start iterative optimization.
    for it in range(iter):
        if(it == 0):
            theta0 = theta.copy()
        # Update the kernel weights.
        else:
            theta0 = theta.copy()
            new_theta = []
            vals = []
            for m in range(P):
                vals.append(WithinClusterSS(K[:,:,m], H))
                # Compute the within-cluster sum of squares for each view.
                new_theta.append(WithinClusterSS(K[:,:,m], H))
            theta = np.array(new_theta)
            theta = theta / np.sqrt(np.sum(theta**2))
        
        # print(theta0)
        # Update the combined kernel matrix.
        Ktheta = np.zeros((N,N))
        for m in range(P):
            Ktheta += theta[m] * K[:, :, m]

        # Update the indicator matrix H.
        eigvals, eigvecs = np.linalg.eigh(Ktheta)
        id = np.argsort(eigvals)[::-1]
        eigvals = eigvals[id]
        eigvecs = eigvecs[:,id]
        H = eigvecs[:,:c]

        # Check for convergence.
        if(np.linalg.norm(theta0 - theta) < epsilon and it > 1):
            break
        if it == iter - 1:
            print(f"The algorithm has not converged after {it + 1} iterations")

    # Recover hard cluster labels.
    kmeans = KMeans(
        n_clusters=c,
        n_init=10,
        random_state=0,
        max_iter=500
    )
    labels = kmeans.fit_predict(H) 

    state = {}
    state['labels'] = labels
    state["theta"] = theta

    return state
