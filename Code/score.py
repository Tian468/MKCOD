import numpy as np
from scipy.spatial.distance import cdist

def get_score(X, model):

    assert isinstance(X, np.ndarray), "X must be a numpy.ndarray"
    assert X.ndim == 2, "X must be a two-dimensional array"
    assert not np.isnan(X).any(), "X contains NaN values"

    centers = model["centers"]
    cluster_sizes = model["cluster_sizes"]
    cluster_sigmas = model["cluster_sigmas"] 
    large_cluster_id = model["large_cluster_id"]
    small_cluster_id = model["small_cluster_id"]
    max_cluster_size = model["max_cluster_size"]

    assert centers.shape[1] == X.shape[1], (
        "Test samples and training samples must have the same number of features"
    )

    N = X.shape[0]
    c = centers.shape[0]

    # Find the nearest cluster center.
    dis_matrix = cdist(X, centers, metric="euclidean")
    belong_to_cluster = np.argmin(dis_matrix, axis=1)

    # Determine whether each sample belongs to a large or small cluster.
    sample_is_large = np.isin(belong_to_cluster, large_cluster_id)
    sample_is_small = np.isin(belong_to_cluster, small_cluster_id)

    # Calculate anomaly scores.
    score = np.zeros(N)
    # Samples assigned to large clusters.
    if np.any(sample_is_large):
        idx = np.where(sample_is_large)[0]
        cluster_idx = belong_to_cluster[idx]

        dist = dis_matrix[idx, cluster_idx]
        sigma = cluster_sigmas[cluster_idx]

        score[idx] = dist / sigma

    # Samples assigned to small clusters.
    if np.any(sample_is_small):
        idx = np.where(sample_is_small)[0]
        cluster_idx = belong_to_cluster[idx]

        large_centers = centers[large_cluster_id]
        dist_to_large_centers = cdist(X[idx], large_centers, metric="euclidean")

        # Find the nearest large cluster.
        nearest_large_pos = np.argmin(dist_to_large_centers, axis=1)
        nearest_large_id = large_cluster_id[nearest_large_pos]

        dist = dist_to_large_centers[np.arange(len(idx)), nearest_large_pos]
        sigma = cluster_sigmas[nearest_large_id]

        # Calculate the cluster rarity factor.
        r = 1.0 + np.log(max_cluster_size / cluster_sizes[cluster_idx])
        score[idx] = (dist / sigma) * r

    return score
