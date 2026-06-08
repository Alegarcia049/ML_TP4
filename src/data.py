from pathlib import Path
import numpy as np
import pandas as pd


def load_fashion_mnist_csv(
    file_path: str | Path,
    label_col: str | None = None,
    normalize_pixels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads Fashion-MNIST data from a CSV file and returns features and labels.

    The label column should be explicitly provided when the CSV structure is ambiguous.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The CSV file is empty.")

    label_col = _infer_label_column(df, label_col)

    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col]).to_numpy()

    X = X.astype(np.float32)
    y = y.astype(int)

    validate_fashion_mnist_arrays(X, y)

    if normalize_pixels:
        X = normalize_pixel_values(X)

    return X, y


def _infer_label_column(df: pd.DataFrame, label_col: str | None) -> str:
    """
    Infers the label column only when a standard label name exists.
    """
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' was not found in the CSV.")
        return label_col

    common_names = ["label", "labels", "class", "target", "y"]

    for col in common_names:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not infer the label column safely. "
        "Use preview_csv_columns(file_path) and pass label_col explicitly."
    )



def validate_fashion_mnist_arrays(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validates basic Fashion-MNIST array shapes and values.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")

    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array. Got shape {y.shape}.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got {X.shape[0]} and {y.shape[0]}."
        )

    if X.shape[1] != 784:
        raise ValueError(
            f"Expected 784 pixel features for 28x28 images. Got {X.shape[1]}."
        )

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or infinite values.")


def normalize_pixel_values(X: np.ndarray) -> np.ndarray:
    """
    Scales pixel values to [0, 1] if they are in [0, 255].
    """
    x_min = X.min()
    x_max = X.max()

    if x_min < 0:
        raise ValueError("Pixel values must be non-negative.")

    if x_max > 1.0:
        return X / 255.0

    return X


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.8,
    random_state: int = 42,
    shuffle: bool = True,
    return_indices: bool = False,
):
    """
    Splits X and y into train and test sets preserving class proportions.

    If return_indices=True, also returns the original train/test indices.
    """
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        cls_indices = rng.permutation(cls_indices)

        n_cls = len(cls_indices)
        n_train = int(np.floor(train_size * n_cls))

        if n_cls > 1:
            n_train = max(1, min(n_train, n_cls - 1))

        train_indices.append(cls_indices[:n_train])
        test_indices.append(cls_indices[n_train:])

    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)

    if shuffle:
        train_indices = rng.permutation(train_indices)
        test_indices = rng.permutation(test_indices)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    if return_indices:
        return X_train, X_test, y_train, y_test, train_indices, test_indices

    return X_train, X_test, y_train, y_test

def stratified_sample_indices(
    y: np.ndarray,
    n_samples: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Returns stratified sample indices preserving class proportions approximately.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if n_samples > len(y):
        raise ValueError("n_samples cannot be larger than the dataset size.")

    rng = np.random.default_rng(random_state)

    labels, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()

    raw_allocations = proportions * n_samples
    base_allocations = np.floor(raw_allocations).astype(int)

    remaining = n_samples - base_allocations.sum()
    fractional_parts = raw_allocations - base_allocations

    # Assign remaining samples to classes with largest fractional parts.
    if remaining > 0:
        order = np.argsort(fractional_parts)[::-1]
        for idx in order[:remaining]:
            base_allocations[idx] += 1

    sampled_indices = []

    for label, n_label_samples in zip(labels, base_allocations):
        label_indices = np.where(y == label)[0]

        if n_label_samples > len(label_indices):
            n_label_samples = len(label_indices)

        selected = rng.choice(
            label_indices,
            size=n_label_samples,
            replace=False,
        )

        sampled_indices.append(selected)

    sampled_indices = np.concatenate(sampled_indices)
    sampled_indices = rng.permutation(sampled_indices)

    return sampled_indices


def stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    random_state: int = 42,
    return_indices: bool = False,
):
    """
    Returns a stratified sample from X and y.
    """
    indices = stratified_sample_indices(
        y=y,
        n_samples=n_samples,
        random_state=random_state,
    )

    if return_indices:
        return X[indices], y[indices], indices

    return X[indices], y[indices]


def class_distribution(y: np.ndarray) -> pd.DataFrame:
    """
    Computes class counts and proportions.
    """
    labels, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()

    return pd.DataFrame(
        {
            "label": labels,
            "count": counts,
            "proportion": proportions,
        }
    )


def compare_split_distribution(
    y_full: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Compares class proportions before and after a train/test split.
    """
    full = class_distribution(y_full).set_index("label")
    train = class_distribution(y_train).set_index("label")
    test = class_distribution(y_test).set_index("label")

    comparison = pd.DataFrame(index=full.index)
    comparison["full_count"] = full["count"]
    comparison["train_count"] = train["count"]
    comparison["test_count"] = test["count"]
    comparison["full_prop"] = full["proportion"]
    comparison["train_prop"] = train["proportion"]
    comparison["test_prop"] = test["proportion"]

    return comparison.reset_index()


def preview_csv_columns(file_path: str | Path, n_columns: int = 8) -> pd.Index:
    """
    Prints and returns the first CSV column names to inspect the dataset structure.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    columns = pd.read_csv(file_path, nrows=0).columns
    print("First columns:", list(columns[:n_columns]))
    return columns