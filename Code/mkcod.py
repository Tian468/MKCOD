import numpy as np

from kernel import get_kernel
from mkkc import mkkcEst
from model import create_model
from score import get_score


class MKCOD:
    def __init__(self, n_clusters=10, alpha=0.9, rbf_gamma_list=None, poly_params=None, use_linear=False, iter=50, epsilon=1e-4):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.rbf_gamma_list = rbf_gamma_list
        self.poly_params = poly_params
        self.use_linear = use_linear
        self.iter = iter
        self.epsilon = epsilon

        self.model_ = None
        self.labels_ = None
        self.theta_ = None

    def _check_X(self, X):
        X = np.asarray(X)
        assert X.ndim == 2, "X must be a two-dimensional array"
        assert not np.isnan(X).any(), "X contains NaN values"
        return X

    def fit(self, X, y=None):
        X = self._check_X(X)
        assert self.n_clusters > 0 and int(self.n_clusters) == self.n_clusters, (
            "n_clusters must be a positive integer"
        )
        assert 0 < self.alpha <= 1, "alpha must be in (0, 1]"

        kernels = get_kernel(
            X,
            self.rbf_gamma_list,
            self.poly_params,
            self.use_linear,
        )
        cluster_state = mkkcEst(
            kernels,
            c=int(self.n_clusters),
            iter=self.iter,
            epsilon=self.epsilon,
        )
        self.labels_ = cluster_state["labels"]
        self.theta_ = cluster_state["theta"]
        self.model_ = create_model(X, self.labels_, alpha=self.alpha)

        return self

    def decision_function(self, X):
        if self.model_ is None:
            raise RuntimeError("Call fit(X_train) first")

        X = self._check_X(X)
        return get_score(X, self.model_)
