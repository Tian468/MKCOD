import numpy as np
from scipy.spatial.distance import cdist


def create_model(X, label_cluster, alpha=0.9):
    """Build the cluster statistics used to calculate anomaly scores."""
    assert 0 < alpha <= 1, "alpha must be in (0, 1]"
    assert isinstance(X, np.ndarray) and X.ndim == 2, (
        "X must be a two-dimensional array"
    )
    assert not np.isnan(X).any(), "X contains NaN values"
    assert label_cluster.shape[0] == X.shape[0], (
        "label_cluster must contain one label per sample"
    )

    n_samples = X.shape[0]
    n_clusters = len(np.unique(label_cluster))
    epsilon = 1e-6

    # Calculate cluster centers.
    centers = np.zeros((n_clusters, X.shape[1]))
    for cluster_id in range(n_clusters):
        mask = label_cluster == cluster_id
        if np.any(mask):
            centers[cluster_id] = X[mask].mean(axis=0)
        else:
            raise ValueError(f"Cluster {cluster_id} contains no samples")

    # Calculate the average distance from samples to their own cluster center.
    all_distances = cdist(X, centers, metric="euclidean")
    own_center_distances = all_distances[np.arange(n_samples), label_cluster]
    cluster_sigmas = np.zeros(n_clusters)
    for cluster_id in range(n_clusters):
        mask = label_cluster == cluster_id
        cluster_sigmas[cluster_id] = own_center_distances[mask].mean()
    cluster_sigmas = np.maximum(cluster_sigmas, epsilon)

    # Divide clusters into large and small groups by cumulative sample ratio.
    cluster_sizes = np.bincount(label_cluster, minlength=n_clusters)
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    cumulative_ratio = np.cumsum(cluster_sizes[sorted_indices]) / n_samples
    split_index = min(np.searchsorted(cumulative_ratio, alpha), n_clusters - 1)
    large_cluster_id = sorted_indices[:split_index + 1]
    small_cluster_id = sorted_indices[split_index + 1:]

    return {
        "centers": centers,
        "cluster_sizes": cluster_sizes,
        "cluster_sigmas": cluster_sigmas,
        "large_cluster_id": large_cluster_id,
        "small_cluster_id": small_cluster_id,
        "max_cluster_size": cluster_sizes.max(),
    }
