import numpy as np


def fit_standardizer(X_train: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes feature-wise mean and standard deviation using only training data.

    Parameters
    ----------
    X_train:
        Training feature matrix with shape (n_samples, n_features).
    eps:
        Small threshold used to detect near-zero standard deviations.

    Returns
    -------
    mean:
        Feature-wise mean.
    std:
        Feature-wise standard deviation with safe values for zero-variance features.
    """
    _validate_2d_array(X_train, "X_train")

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std_safe = np.where(std < eps, 1.0, std)

    return mean.astype(np.float32), std_safe.astype(np.float32)


def transform_standardizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Standardizes data using precomputed training mean and standard deviation.
    """
    _validate_2d_array(X, "X")
    _validate_vector(mean, "mean")
    _validate_vector(std, "std")

    if X.shape[1] != mean.shape[0] or X.shape[1] != std.shape[0]:
        raise ValueError("X, mean and std have incompatible shapes.")

    return ((X - mean) / std).astype(np.float32)


def inverse_transform_standardizer(
    X_scaled: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Reverts a standardization transformation.
    """
    _validate_2d_array(X_scaled, "X_scaled")
    _validate_vector(mean, "mean")
    _validate_vector(std, "std")

    if X_scaled.shape[1] != mean.shape[0] or X_scaled.shape[1] != std.shape[0]:
        raise ValueError("X_scaled, mean and std have incompatible shapes.")

    return (X_scaled * std + mean).astype(np.float32)


def fit_transform_standardizer(
    X_train: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits the standardizer on training data and returns the transformed data.
    """
    mean, std = fit_standardizer(X_train, eps=eps)
    X_train_scaled = transform_standardizer(X_train, mean, std)

    return X_train_scaled, mean, std


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


def _validate_vector(x: np.ndarray, name: str) -> None:
    """
    Validates that an array is one-dimensional and finite.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")

    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D array. Got shape {x.shape}.")

    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinite values.")