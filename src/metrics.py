import numpy as np
import pandas as pd


def pairwise_distances(
    X: np.ndarray,
    squared: bool = False,
    chunk_size: int = 512,
) -> np.ndarray:
    """
    Computes pairwise Euclidean distances.

    For n=3000, the full matrix has 9M entries. Using float32, this is
    approximately 36 MB.
    """
    _validate_2d_array(X, "X")

    n_samples = X.shape[0]
    distances = np.empty((n_samples, n_samples), dtype=np.float32)

    X_norm = np.sum(X**2, axis=1)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)

        block = (
            X_norm[start:end, np.newaxis]
            + X_norm[np.newaxis, :]
            - 2.0 * X[start:end] @ X.T
        )

        block = np.maximum(block, 0.0)

        if not squared:
            block = np.sqrt(block)

        distances[start:end] = block.astype(np.float32)

    return distances


def silhouette_score_custom(
    X: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    distances: np.ndarray | None = None,
) -> float:
    """
    Computes the mean Silhouette score.

    Either X or a precomputed distance matrix must be provided.
    """
    if labels is None:
        raise ValueError("labels must be provided.")

    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")

    if distances is None:
        if X is None:
            raise ValueError("Either X or distances must be provided.")
        distances = pairwise_distances(X)

    if distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix.")

    if distances.shape[0] != labels.shape[0]:
        raise ValueError("distances and labels have incompatible shapes.")

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    n_samples = labels.shape[0]
    silhouette_values = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        same_cluster_mask = labels == labels[i]
        same_cluster_count = same_cluster_mask.sum()

        if same_cluster_count <= 1:
            silhouette_values[i] = 0.0
            continue

        a_i = distances[i, same_cluster_mask].sum() / (same_cluster_count - 1)

        b_i = np.inf

        for other_label in unique_labels:
            if other_label == labels[i]:
                continue

            other_cluster_mask = labels == other_label
            mean_distance = distances[i, other_cluster_mask].mean()

            if mean_distance < b_i:
                b_i = mean_distance

        denominator = max(a_i, b_i)

        if denominator > 0:
            silhouette_values[i] = (b_i - a_i) / denominator
        else:
            silhouette_values[i] = 0.0

    return float(np.mean(silhouette_values))


def cluster_size_table(labels: np.ndarray) -> pd.DataFrame:
    """
    Returns cluster sizes.
    """
    labels = np.asarray(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)

    return pd.DataFrame(
        {
            "cluster": unique_labels,
            "size": counts,
            "proportion": counts / counts.sum(),
        }
    )


def cluster_class_table(labels: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """
    Returns a cluster-by-class contingency table.
    """
    labels = np.asarray(labels)
    y_true = np.asarray(y_true)

    if labels.shape[0] != y_true.shape[0]:
        raise ValueError("labels and y_true must have the same length.")

    table = pd.crosstab(
        pd.Series(labels, name="cluster"),
        pd.Series(y_true, name="class"),
    )

    return table.reset_index()


def cluster_purity_table(labels: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """
    Computes cluster-level purity using true labels only for interpretation.

    Purity must not be used for model selection because it uses ground-truth labels.
    """
    contingency = cluster_class_table(labels, y_true).set_index("cluster")

    cluster_sizes = contingency.sum(axis=1)
    majority_counts = contingency.max(axis=1)
    majority_classes = contingency.idxmax(axis=1)
    purity = majority_counts / cluster_sizes

    return pd.DataFrame(
        {
            "cluster": contingency.index,
            "size": cluster_sizes.values,
            "majority_class": majority_classes.values,
            "majority_count": majority_counts.values,
            "purity": purity.values,
        }
    )


def add_marginal_gain(
    metrics_df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
    mode: str,
    output_col: str = "marginal_gain",
) -> pd.DataFrame:
    """
    Adds marginal gain by K within each group.

    mode='decrease': gain = previous_metric - current_metric.
    mode='increase': gain = current_metric - previous_metric.
    """
    if mode not in {"decrease", "increase"}:
        raise ValueError("mode must be either 'decrease' or 'increase'.")

    df = metrics_df.copy()
    df = df.sort_values(group_cols + ["K"])

    df[output_col] = np.nan
    df[f"{output_col}_relative"] = np.nan

    for _, group_index in df.groupby(group_cols).groups.items():
        group = df.loc[group_index].sort_values("K")
        values = group[metric_col].to_numpy(dtype=float)

        gains = np.full_like(values, fill_value=np.nan, dtype=float)
        relative_gains = np.full_like(values, fill_value=np.nan, dtype=float)

        for i in range(1, len(values)):
            if mode == "decrease":
                gain = values[i - 1] - values[i]
            else:
                gain = values[i] - values[i - 1]

            gains[i] = gain

            denominator = abs(values[i - 1])
            if denominator > 0:
                relative_gains[i] = gain / denominator

        df.loc[group.index, output_col] = gains
        df.loc[group.index, f"{output_col}_relative"] = relative_gains

    return df


def overall_purity(labels: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes overall clustering purity.
    """
    purity_df = cluster_purity_table(labels, y_true)
    return float(purity_df["majority_count"].sum() / purity_df["size"].sum())


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