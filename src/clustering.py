import numpy as np


class KMeansCustom:
    """
    k-Means implementation using numpy.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
        init: str = "kmeans++",
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        if init not in {"random", "kmeans++"}:
            raise ValueError("init must be either 'random' or 'kmeans++'.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def fit(self, X: np.ndarray) -> "KMeansCustom":
        """
        Fits k-Means to X.
        """
        _validate_2d_array(X, "X")

        rng = np.random.default_rng(self.random_state)
        centers = self._initialize_centers(X, rng)

        for iteration in range(1, self.max_iter + 1):
            distances = _squared_euclidean_distances(X, centers)
            labels = np.argmin(distances, axis=1)

            new_centers = self._compute_centers(X, labels, centers, rng)

            shift = np.sqrt(np.sum((new_centers - centers) ** 2))
            centers = new_centers

            if shift <= self.tol:
                break

        distances = _squared_euclidean_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances, axis=1))

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(inertia)
        self.n_iter_ = iteration

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assigns each sample to the nearest fitted center.
        """
        _validate_2d_array(X, "X")

        if self.cluster_centers_ is None:
            raise ValueError("The model must be fitted before prediction.")

        distances = _squared_euclidean_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the model and returns cluster labels.
        """
        self.fit(X)
        return self.labels_

    def _initialize_centers(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Initializes cluster centers.
        """
        n_samples = X.shape[0]

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples.")

        if self.init == "random":
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()

        return self._initialize_kmeans_plus_plus(X, rng)

    def _initialize_kmeans_plus_plus(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Initializes centers with the k-means++ strategy.
        """
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=X.dtype)

        first_idx = rng.integers(0, n_samples)
        centers[0] = X[first_idx]

        closest_dist_sq = _squared_euclidean_distances(X, centers[[0]]).reshape(-1)

        for center_idx in range(1, self.n_clusters):
            total_dist = closest_dist_sq.sum()

            if total_dist <= 0:
                candidate_idx = rng.integers(0, n_samples)
            else:
                probabilities = closest_dist_sq / total_dist
                candidate_idx = rng.choice(n_samples, p=probabilities)

            centers[center_idx] = X[candidate_idx]

            new_dist_sq = _squared_euclidean_distances(
                X,
                centers[[center_idx]],
            ).reshape(-1)

            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centers

    def _compute_centers(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        old_centers: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Computes new centers and handles empty clusters.
        """
        centers = np.empty_like(old_centers)

        distances_to_old = _squared_euclidean_distances(X, old_centers)
        min_distances = np.min(distances_to_old, axis=1)

        for cluster_idx in range(self.n_clusters):
            mask = labels == cluster_idx

            if np.any(mask):
                centers[cluster_idx] = X[mask].mean(axis=0)
            else:
                # Reinitialize empty cluster with a point far from its nearest center.
                farthest_idx = np.argmax(min_distances)
                centers[cluster_idx] = X[farthest_idx]
                min_distances[farthest_idx] = 0.0

        return centers


class GaussianMixtureCustom:
    """
    Gaussian Mixture Model using EM and diagonal covariance matrices.
    """

    def __init__(
        self,
        n_components: int,
        covariance_type: str = "diag",
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int = 42,
        init_params: str = "kmeans",
    ) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive.")

        if covariance_type != "diag":
            raise NotImplementedError("Only covariance_type='diag' is implemented.")

        if init_params not in {"kmeans", "random"}:
            raise ValueError("init_params must be either 'kmeans' or 'random'.")

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init_params = init_params

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.labels_ = None
        self.log_likelihood_history_ = []
        self.n_iter_ = 0

    def fit(self, X: np.ndarray) -> "GaussianMixtureCustom":
        """
        Fits the GMM using the EM algorithm.
        """
        _validate_2d_array(X, "X")

        self._initialize_parameters(X)

        previous_log_likelihood = -np.inf

        for iteration in range(1, self.max_iter + 1):
            log_resp, log_prob_norm = self._e_step(X)
            responsibilities = np.exp(log_resp)

            self._m_step(X, responsibilities)

            avg_log_likelihood = float(np.mean(log_prob_norm))
            self.log_likelihood_history_.append(avg_log_likelihood)

            improvement = avg_log_likelihood - previous_log_likelihood

            if iteration > 1 and abs(improvement) <= self.tol:
                break

            previous_log_likelihood = avg_log_likelihood

        self.n_iter_ = iteration
        self.labels_ = self.predict(X)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns posterior responsibilities for each component.
        """
        _validate_2d_array(X, "X")

        if self.means_ is None:
            raise ValueError("The model must be fitted before prediction.")

        log_resp, _ = self._e_step(X)
        return np.exp(log_resp)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the most likely component for each sample.
        """
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the model and returns cluster labels.
        """
        self.fit(X)
        return self.labels_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns log likelihood per sample.
        """
        _validate_2d_array(X, "X")

        weighted_log_prob = self._estimate_weighted_log_prob(X)
        return _logsumexp(weighted_log_prob, axis=1)

    def score(self, X: np.ndarray) -> float:
        """
        Returns average log likelihood.
        """
        return float(np.mean(self.score_samples(X)))

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initializes weights, means and diagonal covariances.
        """
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        if self.n_components > n_samples:
            raise ValueError("n_components cannot exceed number of samples.")

        if self.init_params == "kmeans":
            kmeans = KMeansCustom(
                n_clusters=self.n_components,
                max_iter=50,
                random_state=self.random_state,
                init="kmeans++",
            )
            labels = kmeans.fit_predict(X)
            means = kmeans.cluster_centers_
        else:
            indices = rng.choice(n_samples, size=self.n_components, replace=False)
            means = X[indices].copy()
            labels = rng.integers(0, self.n_components, size=n_samples)

        weights = np.zeros(self.n_components, dtype=np.float64)
        covariances = np.zeros((self.n_components, n_features), dtype=np.float64)

        global_var = np.var(X, axis=0) + self.reg_covar

        for component_idx in range(self.n_components):
            mask = labels == component_idx

            if np.any(mask):
                weights[component_idx] = mask.mean()
                covariances[component_idx] = np.var(X[mask], axis=0) + self.reg_covar
            else:
                weights[component_idx] = 1.0 / self.n_components
                covariances[component_idx] = global_var

        weights = weights / weights.sum()

        self.weights_ = weights
        self.means_ = means.astype(np.float64)
        self.covariances_ = covariances.astype(np.float64)

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the E-step.
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = _logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        return log_resp, log_prob_norm

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        Runs the M-step.
        """
        n_samples, n_features = X.shape
        eps = 1e-12

        nk = responsibilities.sum(axis=0) + eps
        weights = nk / n_samples

        means = (responsibilities.T @ X) / nk[:, np.newaxis]

        covariances = np.empty((self.n_components, n_features), dtype=np.float64)

        for component_idx in range(self.n_components):
            diff = X - means[component_idx]
            weighted_diff = responsibilities[:, component_idx][:, np.newaxis] * diff**2
            covariances[component_idx] = (
                weighted_diff.sum(axis=0) / nk[component_idx]
            ) + self.reg_covar

        weights = np.clip(weights, eps, None)
        weights = weights / weights.sum()

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances

    def _estimate_weighted_log_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Computes log p(x, z=k) for all samples and components.
        """
        log_prob = _estimate_log_gaussian_diag(
            X,
            self.means_,
            self.covariances_,
        )

        log_weights = np.log(np.clip(self.weights_, 1e-12, None))
        return log_prob + log_weights


def _squared_euclidean_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Computes squared Euclidean distances from samples to centers.
    """
    X_norm = np.sum(X**2, axis=1, keepdims=True)
    centers_norm = np.sum(centers**2, axis=1)
    distances = X_norm + centers_norm - 2.0 * X @ centers.T
    return np.maximum(distances, 0.0)


def _estimate_log_gaussian_diag(
    X: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """
    Computes log Gaussian probabilities for diagonal covariance matrices.
    """
    n_features = X.shape[1]

    log_det = np.sum(np.log(covariances), axis=1)
    precisions = 1.0 / covariances

    log_prob = np.empty((X.shape[0], means.shape[0]), dtype=np.float64)

    for component_idx in range(means.shape[0]):
        diff = X - means[component_idx]
        mahalanobis = np.sum(diff**2 * precisions[component_idx], axis=1)

        log_prob[:, component_idx] = -0.5 * (
            n_features * np.log(2.0 * np.pi)
            + log_det[component_idx]
            + mahalanobis
        )

    return log_prob


def _logsumexp(X: np.ndarray, axis: int) -> np.ndarray:
    """
    Stable log-sum-exp implementation.
    """
    max_value = np.max(X, axis=axis, keepdims=True)
    stable_sum = np.sum(np.exp(X - max_value), axis=axis, keepdims=True)
    return (max_value + np.log(stable_sum)).squeeze(axis=axis)


def _validate_2d_array(X: np.ndarray, name: str) -> None:
    """
    Validates that X is a finite 2D numpy array.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")

    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape {X.shape}.")

    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinite values.")