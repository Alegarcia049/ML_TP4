import numpy as np
import pandas as pd


def fit_pca(
    X_scaled: np.ndarray,
    center: bool = False,
) -> dict[str, np.ndarray]:
    """
    Fits PCA using Singular Value Decomposition.

    Parameters
    ----------
    X_scaled:
        Standardized feature matrix with shape (n_samples, n_features).
        If center=False, this array is assumed to be already centered.
    center:
        If True, centers X_scaled inside PCA. For this project, use False
        because standardization was already fitted on training data.

    Returns
    -------
    pca_result:
        Dictionary containing PCA parameters and variance information.
    """
    _validate_2d_array(X_scaled, "X_scaled")

    n_samples = X_scaled.shape[0]

    if n_samples < 2:
        raise ValueError("PCA requires at least two samples.")

    if center:
        mean = np.mean(X_scaled, axis=0)
        X_centered = X_scaled - mean
    else:
        mean = None
        X_centered = X_scaled

    # SVD: X = U S Vt. Rows of Vt are principal directions.
    _, singular_values, components = np.linalg.svd(X_centered, full_matrices=False)

    eigenvalues = (singular_values ** 2) / (n_samples - 1)

    total_variance = np.sum(eigenvalues)

    if total_variance <= 0:
        raise ValueError("Total variance must be positive.")

    explained_variance_ratio = eigenvalues / total_variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    return {
        "components": components.astype(np.float32),
        "singular_values": singular_values.astype(np.float32),
        "eigenvalues": eigenvalues.astype(np.float32),
        "explained_variance_ratio": explained_variance_ratio.astype(np.float32),
        "cumulative_explained_variance": cumulative_explained_variance.astype(np.float32),
        "mean": None if mean is None else mean.astype(np.float32),
    }


def transform_pca(
    X_scaled: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray | None = None,
    n_components: int | None = None,
) -> np.ndarray:
    """
    Projects standardized data into PCA latent space.
    """
    _validate_2d_array(X_scaled, "X_scaled")
    _validate_2d_array(components, "components")

    if X_scaled.shape[1] != components.shape[1]:
        raise ValueError("X_scaled and components have incompatible shapes.")

    if mean is not None:
        if mean.ndim != 1 or mean.shape[0] != X_scaled.shape[1]:
            raise ValueError("mean has incompatible shape.")
        X_centered = X_scaled - mean
    else:
        X_centered = X_scaled

    if n_components is not None:
        if n_components <= 0 or n_components > components.shape[0]:
            raise ValueError("n_components is out of valid range.")
        components = components[:n_components]

    return (X_centered @ components.T).astype(np.float32)


def inverse_transform_pca(
    Z: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reconstructs standardized data from PCA latent representation.
    """
    _validate_2d_array(Z, "Z")
    _validate_2d_array(components, "components")

    if Z.shape[1] != components.shape[0]:
        raise ValueError("Z and components have incompatible shapes.")

    X_reconstructed = Z @ components

    if mean is not None:
        if mean.ndim != 1 or mean.shape[0] != components.shape[1]:
            raise ValueError("mean has incompatible shape.")
        X_reconstructed = X_reconstructed + mean

    return X_reconstructed.astype(np.float32)


def select_n_components(
    cumulative_explained_variance: np.ndarray,
    threshold: float = 0.90,
) -> int:
    """
    Selects the minimum number of components needed to reach a variance threshold.
    """
    if cumulative_explained_variance.ndim != 1:
        raise ValueError("cumulative_explained_variance must be a 1D array.")

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in the interval (0, 1].")

    if cumulative_explained_variance[-1] < threshold:
        raise ValueError("The threshold cannot be reached with the provided components.")

    return int(np.searchsorted(cumulative_explained_variance, threshold) + 1)


def make_pca_variance_table(
    explained_variance_ratio: np.ndarray,
    cumulative_explained_variance: np.ndarray,
) -> pd.DataFrame:
    """
    Builds a PCA variance summary table.
    """
    if explained_variance_ratio.shape != cumulative_explained_variance.shape:
        raise ValueError("Variance arrays must have the same shape.")

    return pd.DataFrame(
        {
            "component": np.arange(1, len(explained_variance_ratio) + 1),
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance": cumulative_explained_variance,
        }
    )


def reconstruction_mse(X_original: np.ndarray, X_reconstructed: np.ndarray) -> float:
    """
    Computes mean squared reconstruction error.
    """
    if X_original.shape != X_reconstructed.shape:
        raise ValueError("X_original and X_reconstructed must have the same shape.")

    return float(np.mean((X_original - X_reconstructed) ** 2))


def _validate_2d_array(X: np.ndarray, name: str) -> None:
    """
    Validates that an array is two-dimensional and finite.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")

    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape {X.shape}.")

    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinite values.")