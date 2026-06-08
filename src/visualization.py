import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def _save_show_or_close(
    fig,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Saves, shows, or closes a matplotlib figure.
    """
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.15,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_multiple_reconstructions(
    X_original: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    indices: np.ndarray | list[int] | None = None,
    n_images: int = 10,
    image_shape: tuple[int, int] = (28, 28),
    title: str = "Reconstruction comparison",
    cmap: str = "gray",
    clip_values: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots original images and multiple reconstruction sets.

    The first row contains original images. Each following row contains one
    reconstruction method. Row labels are placed with axis-level text and
    extra margin to avoid clipping when saving.
    """
    if X_original.ndim != 2:
        raise ValueError("X_original must have shape (n_samples, n_features).")

    if len(reconstructions) == 0:
        raise ValueError("At least one reconstruction set must be provided.")

    for name, X_rec in reconstructions.items():
        if X_rec.shape != X_original.shape:
            raise ValueError(
                f"Reconstruction '{name}' has shape {X_rec.shape}, "
                f"expected {X_original.shape}."
            )

    if X_original.shape[1] != image_shape[0] * image_shape[1]:
        raise ValueError("Image shape is incompatible with number of features.")

    if indices is None:
        indices = np.arange(min(n_images, X_original.shape[0]))
    else:
        indices = np.array(indices)
        n_images = len(indices)

    row_labels = ["Original"] + list(reconstructions.keys())
    n_rows = len(row_labels)

    fig, axes = plt.subplots(
        n_rows,
        n_images,
        figsize=(2.1 * n_images, 2.2 * n_rows),
    )

    axes = np.array(axes).reshape(n_rows, n_images)

    for col_idx, sample_idx in enumerate(indices):
        image = X_original[sample_idx].reshape(image_shape)

        if clip_values:
            image = np.clip(image, 0, 1)

        axes[0, col_idx].imshow(image, cmap=cmap)
        axes[0, col_idx].axis("off")
        axes[0, col_idx].set_title(f"Idx {sample_idx}", fontsize=10)

    for row_idx, (_, X_rec) in enumerate(reconstructions.items(), start=1):
        for col_idx, sample_idx in enumerate(indices):
            image = X_rec[sample_idx].reshape(image_shape)

            if clip_values:
                image = np.clip(image, 0, 1)

            axes[row_idx, col_idx].imshow(image, cmap=cmap)
            axes[row_idx, col_idx].axis("off")

    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(
            -0.35,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )

    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.90, bottom=0.04)

    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_vae_training_curves(
    history: dict,
    title: str = "beta-VAE training curves",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots beta-VAE total, reconstruction and KL losses for train and validation.
    """
    required_keys = [
        "epoch",
        "train_total_loss",
        "train_reconstruction_loss",
        "train_kl_loss",
        "val_total_loss",
        "val_reconstruction_loss",
        "val_kl_loss",
    ]

    for key in required_keys:
        if key not in history:
            raise ValueError(f"Missing key in history: {key}")

    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_total_loss"], label="Train")
    axes[0].plot(epochs, history["val_total_loss"], label="Validation")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_reconstruction_loss"], label="Train")
    axes[1].plot(epochs, history["val_reconstruction_loss"], label="Validation")
    axes[1].set_title("Reconstruction loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE sum per sample")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, history["train_kl_loss"], label="Train")
    axes[2].plot(epochs, history["val_kl_loss"], label="Validation")
    axes[2].set_title("KL loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Mean KL per latent dimension")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_latent_average_comparison(
    X_reference: np.ndarray,
    decoded_images: dict[str, np.ndarray],
    image_shape: tuple[int, int] = (28, 28),
    title: str = "Latent average comparison",
    cmap: str = "gray",
    clip_values: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots reference images and decoded average latent representations.
    """
    if X_reference.ndim != 2:
        raise ValueError("X_reference must have shape (n_samples, n_features).")

    if len(decoded_images) == 0:
        raise ValueError("decoded_images cannot be empty.")

    n_reference = X_reference.shape[0]
    n_decoded = len(decoded_images)
    n_cols = n_reference + n_decoded

    fig, axes = plt.subplots(1, n_cols, figsize=(2.3 * n_cols, 2.7))
    axes = np.array(axes).reshape(-1)

    for idx in range(n_reference):
        image = X_reference[idx].reshape(image_shape)

        if clip_values:
            image = np.clip(image, 0, 1)

        axes[idx].imshow(image, cmap=cmap)
        axes[idx].axis("off")
        axes[idx].set_title(f"Ref {idx + 1}")

    start_idx = n_reference

    for offset, (name, decoded) in enumerate(decoded_images.items()):
        decoded = np.asarray(decoded)

        if decoded.ndim == 2:
            decoded = decoded[0]

        image = decoded.reshape(image_shape)

        if clip_values:
            image = np.clip(image, 0, 1)

        ax_idx = start_idx + offset
        axes[ax_idx].imshow(image, cmap=cmap)
        axes[ax_idx].axis("off")
        axes[ax_idx].set_title(name)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_random_latent_samples_comparison(
    decoded_samples: dict[str, np.ndarray],
    n_images: int = 10,
    image_shape: tuple[int, int] = (28, 28),
    title: str = "Random latent sampling comparison",
    cmap: str = "gray",
    clip_values: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots decoded samples from random latent vectors for multiple models.
    """
    if len(decoded_samples) == 0:
        raise ValueError("decoded_samples cannot be empty.")

    row_labels = list(decoded_samples.keys())
    n_rows = len(row_labels)

    for name, X_decoded in decoded_samples.items():
        if X_decoded.ndim != 2:
            raise ValueError(f"{name} must have shape (n_samples, n_features).")

        if X_decoded.shape[0] < n_images:
            raise ValueError(f"{name} has fewer samples than n_images.")

    fig, axes = plt.subplots(
        n_rows,
        n_images,
        figsize=(2.1 * n_images, 2.2 * n_rows),
    )

    axes = np.array(axes).reshape(n_rows, n_images)

    for row_idx, (name, X_decoded) in enumerate(decoded_samples.items()):
        for col_idx in range(n_images):
            image = X_decoded[col_idx].reshape(image_shape)

            if clip_values:
                image = np.clip(image, 0, 1)

            axes[row_idx, col_idx].imshow(image, cmap=cmap)
            axes[row_idx, col_idx].axis("off")

            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"z {col_idx + 1}", fontsize=10)

        axes[row_idx, 0].text(
            -0.35,
            0.5,
            name,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )

    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.06)

    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_images(
    X: np.ndarray,
    y: np.ndarray | None = None,
    indices: np.ndarray | list[int] | None = None,
    n_images: int = 15,
    n_cols: int = 5,
    image_shape: tuple[int, int] = (28, 28),
    title: str | None = None,
    cmap: str = "gray",
    random_state: int = 42,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots a selected or random set of images.
    """
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n_samples, n_features). Got {X.shape}.")

    if X.shape[1] != image_shape[0] * image_shape[1]:
        raise ValueError("Image shape is not compatible with the number of features.")

    if indices is None:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(X.shape[0], size=n_images, replace=False)
    else:
        indices = np.array(indices)
        n_images = len(indices)

    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax_idx, sample_idx in enumerate(indices):
        image = X[sample_idx].reshape(image_shape)

        axes[ax_idx].imshow(image, cmap=cmap)
        axes[ax_idx].axis("off")

        if y is not None:
            axes[ax_idx].set_title(f"Label: {y[sample_idx]} | Idx: {sample_idx}")
        else:
            axes[ax_idx].set_title(f"Idx: {sample_idx}")

    for ax in axes[n_images:]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_random_15_images(
    X: np.ndarray,
    y: np.ndarray | None = None,
    random_state: int = 42,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots 15 random images.
    """
    plot_images(
        X=X,
        y=y,
        n_images=15,
        n_cols=5,
        title="15 random Fashion-MNIST samples",
        random_state=random_state,
        save_path=save_path,
        show=show,
    )


def plot_class_distribution(
    y: np.ndarray,
    title: str = "Class distribution",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots the label distribution.
    """
    labels, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=labels, y=counts, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Class label")
    ax.set_ylabel("Count")

    plt.tight_layout()
    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_images_by_class(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[int] | np.ndarray | None = None,
    n_classes: int = 5,
    samples_per_class: int = 5,
    image_shape: tuple[int, int] = (28, 28),
    random_state: int = 42,
    cmap: str = "gray",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots image samples grouped by class.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    rng = np.random.default_rng(random_state)

    available_classes = np.unique(y)

    if classes is None:
        classes = available_classes[:n_classes]
    else:
        classes = np.array(classes)

    n_rows = len(classes)
    n_cols = samples_per_class

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
    )

    axes = np.array(axes).reshape(n_rows, n_cols)

    for row_idx, cls in enumerate(classes):
        cls_indices = np.where(y == cls)[0]

        if len(cls_indices) == 0:
            raise ValueError(f"Class {cls} has no samples.")

        replace = len(cls_indices) < samples_per_class

        selected_indices = rng.choice(
            cls_indices,
            size=samples_per_class,
            replace=replace,
        )

        for col_idx, sample_idx in enumerate(selected_indices):
            image = X[sample_idx].reshape(image_shape)

            axes[row_idx, col_idx].imshow(image, cmap=cmap)
            axes[row_idx, col_idx].axis("off")

            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(
                    f"Class {cls}",
                    fontsize=12,
                    rotation=0,
                    labelpad=35,
                )

    fig.suptitle("Samples grouped by class", fontsize=14)
    plt.tight_layout()
    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_cumulative_explained_variance(
    cumulative_explained_variance: np.ndarray,
    threshold: float = 0.90,
    n_components_selected: int | None = None,
    title: str = "Cumulative explained variance",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots cumulative explained variance as a function of PCA components.
    """
    if cumulative_explained_variance.ndim != 1:
        raise ValueError("cumulative_explained_variance must be a 1D array.")

    components = np.arange(1, len(cumulative_explained_variance) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(components, cumulative_explained_variance, marker="o", markersize=2)
    ax.axhline(threshold, linestyle="--", label=f"{threshold:.0%} threshold")

    if n_components_selected is not None:
        ax.axvline(
            n_components_selected,
            linestyle="--",
            label=f"K = {n_components_selected}",
        )

    ax.set_title(title)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_original_vs_reconstructed(
    X_original: np.ndarray,
    X_reconstructed: np.ndarray,
    indices: np.ndarray | list[int] | None = None,
    n_images: int = 10,
    image_shape: tuple[int, int] = (28, 28),
    title: str = "Original vs reconstructed images",
    cmap: str = "gray",
    clip_values: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots original images and their reconstructions in paired rows.
    """
    if X_original.shape != X_reconstructed.shape:
        raise ValueError("X_original and X_reconstructed must have the same shape.")

    if X_original.ndim != 2:
        raise ValueError("Input arrays must have shape (n_samples, n_features).")

    if X_original.shape[1] != image_shape[0] * image_shape[1]:
        raise ValueError("Image shape is incompatible with the number of features.")

    if indices is None:
        indices = np.arange(min(n_images, X_original.shape[0]))
    else:
        indices = np.array(indices)
        n_images = len(indices)

    fig, axes = plt.subplots(
        2,
        n_images,
        figsize=(2.2 * n_images, 4.5),
    )

    for col_idx, sample_idx in enumerate(indices):
        original = X_original[sample_idx].reshape(image_shape)
        reconstructed = X_reconstructed[sample_idx].reshape(image_shape)

        if clip_values:
            original = np.clip(original, 0, 1)
            reconstructed = np.clip(reconstructed, 0, 1)

        axes[0, col_idx].imshow(original, cmap=cmap)
        axes[0, col_idx].axis("off")
        axes[0, col_idx].set_title(f"Idx {sample_idx}")

        axes[1, col_idx].imshow(reconstructed, cmap=cmap)
        axes[1, col_idx].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("PCA", fontsize=12)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_training_curves(
    history: dict | None = None,
    train_loss: list[float] | np.ndarray | None = None,
    val_loss: list[float] | np.ndarray | None = None,
    title: str = "Training curves",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots train and validation loss curves.
    """
    if history is not None:
        train_loss = history.get("train_loss")
        val_loss = history.get("val_loss")
        epochs = history.get("epoch", np.arange(1, len(train_loss) + 1))
    else:
        if train_loss is None:
            raise ValueError("train_loss must be provided if history is None.")
        epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_loss, label="Train loss")

    if val_loss is not None:
        ax.plot(epochs, val_loss, label="Validation loss")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_multiple_reconstructions(
    X_original: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    indices: np.ndarray | list[int] | None = None,
    n_images: int = 10,
    image_shape: tuple[int, int] = (28, 28),
    title: str = "Reconstruction comparison",
    cmap: str = "gray",
    clip_values: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots original images and multiple reconstruction sets.

    The first row contains original images. Each following row contains one
    reconstruction method. Row labels are placed at figure level to avoid clipping.
    """
    if X_original.ndim != 2:
        raise ValueError("X_original must have shape (n_samples, n_features).")

    if len(reconstructions) == 0:
        raise ValueError("At least one reconstruction set must be provided.")

    for name, X_rec in reconstructions.items():
        if X_rec.shape != X_original.shape:
            raise ValueError(
                f"Reconstruction '{name}' has shape {X_rec.shape}, "
                f"expected {X_original.shape}."
            )

    if X_original.shape[1] != image_shape[0] * image_shape[1]:
        raise ValueError("Image shape is incompatible with number of features.")

    if indices is None:
        indices = np.arange(min(n_images, X_original.shape[0]))
    else:
        indices = np.array(indices)
        n_images = len(indices)

    row_labels = ["Original"] + list(reconstructions.keys())
    n_rows = len(row_labels)

    fig, axes = plt.subplots(
        n_rows,
        n_images,
        figsize=(2.1 * n_images, 2.2 * n_rows),
    )

    axes = np.array(axes).reshape(n_rows, n_images)

    # Row 1: original images.
    for col_idx, sample_idx in enumerate(indices):
        image = X_original[sample_idx].reshape(image_shape)

        if clip_values:
            image = np.clip(image, 0, 1)

        axes[0, col_idx].imshow(image, cmap=cmap)
        axes[0, col_idx].axis("off")
        axes[0, col_idx].set_title(f"Idx {sample_idx}")

    # Remaining rows: reconstructions.
    for row_idx, (_, X_rec) in enumerate(reconstructions.items(), start=1):
        for col_idx, sample_idx in enumerate(indices):
            image = X_rec[sample_idx].reshape(image_shape)

            if clip_values:
                image = np.clip(image, 0, 1)

            axes[row_idx, col_idx].imshow(image, cmap=cmap)
            axes[row_idx, col_idx].axis("off")

    # Figure-level row labels to prevent clipping.
    top = 0.88
    bottom = 0.08

    if n_rows == 1:
        y_positions = [0.5]
    else:
        y_positions = np.linspace(top, bottom, n_rows)

    for label, y_pos in zip(row_labels, y_positions):
        fig.text(
            0.015,
            y_pos,
            label,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=14)

    # Reserve left margin for row labels.
    plt.tight_layout(rect=[0.06, 0.02, 1.0, 0.95])
    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_clustering_metric_by_k(
    metrics_df: pd.DataFrame,
    metric_col: str,
    model_name: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots a clustering metric as a function of K.
    """
    required_cols = {"model", "space", "K", metric_col}

    if not required_cols.issubset(metrics_df.columns):
        raise ValueError(f"metrics_df must contain columns: {required_cols}")

    plot_df = metrics_df.copy()

    if model_name is not None:
        plot_df = plot_df[plot_df["model"] == model_name]

    fig, ax = plt.subplots(figsize=(8, 5))

    for (model, space), group in plot_df.groupby(["model", "space"]):
        group = group.sort_values("K")
        label = f"{model} - {space}"

        ax.plot(
            group["K"],
            group[metric_col],
            marker="o",
            label=label,
        )

    ax.set_title(title or metric_col)
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel or metric_col)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    _save_show_or_close(fig, save_path=save_path, show=show)


def plot_silhouette_vs_k(
    metrics_df: pd.DataFrame,
    model_name: str,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots Silhouette score vs K for a given clustering model.
    """
    plot_clustering_metric_by_k(
        metrics_df=metrics_df,
        metric_col="silhouette_score",
        model_name=model_name,
        title=title or f"{model_name} Silhouette score vs K",
        ylabel="Silhouette score",
        save_path=save_path,
        show=show,
    )


def plot_marginal_gain_vs_k(
    metrics_df: pd.DataFrame,
    model_name: str,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plots marginal gain vs K for a given clustering model.
    """
    plot_clustering_metric_by_k(
        metrics_df=metrics_df,
        metric_col="marginal_gain",
        model_name=model_name,
        title=title or f"{model_name} marginal gain vs K",
        ylabel="Marginal gain",
        save_path=save_path,
        show=show,
    )