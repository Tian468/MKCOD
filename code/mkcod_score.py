import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def mckod_score(H, alpha=0.75, eps=1e-6):
    """
    alpha: cumulative proportion threshold for large clusters

    return:
        score: (N,) anomaly scores
        labels: (N,) cluster labels
    """
    N, c = H.shape

    # KMeans clustering
    kmeans = KMeans(
        n_clusters=c,
        n_init=10,
        random_state=0,
        max_iter=500
    ).fit(H)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Cluster sizes
    cluster_sizes = np.bincount(labels, minlength=c)

    # Distance from each sample to all centers
    dist_all = cdist(H, centers)
    dist_own = dist_all[np.arange(N), labels]

    # Cluster sigma
    cluster_deviation = np.array([
        dist_own[labels == i].mean() if cluster_sizes[i] > 0 else eps
        for i in range(c)
    ])
    cluster_deviation = np.maximum(cluster_deviation, eps)

    # Large / Small cluster split 
    sorted_idx = np.argsort(cluster_sizes)[::-1]
    cum_ratio = np.cumsum(cluster_sizes[sorted_idx]) / N
    cutoff = np.searchsorted(cum_ratio, alpha)

    large_ids = sorted_idx[:cutoff+1]
    is_large = np.isin(labels, large_ids)

    max_size = cluster_sizes[sorted_idx[0]]

    # Map small clusters to nearest large cluster 
    small_ids = np.setdiff1d(np.arange(c), large_ids)

    small_to_large = np.zeros(c, dtype=int)
    if len(small_ids) > 0:
        dist_centers = cdist(centers[small_ids], centers[large_ids])
        nearest = large_ids[np.argmin(dist_centers, axis=1)]
        small_to_large[small_ids] = nearest

    # Compute anomaly scores
    score = np.zeros(N)

    # Samples in large clusters
    score[is_large] = (
        dist_own[is_large] /
        cluster_deviation[labels[is_large]]
    ) 

    # Samples in small clusters
    if len(small_ids) > 0:
        idx = np.where(~is_large)[0]
        small_cluster = labels[idx]
        mapped_large = small_to_large[small_cluster]

        rarity = 1.0 + np.log(max_size / cluster_sizes[small_cluster])

        score[idx] = (
            dist_all[idx, mapped_large] /
            cluster_deviation[mapped_large]
        ) *  rarity

    return score

