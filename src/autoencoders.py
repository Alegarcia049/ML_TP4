from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DeterministicAutoencoder(nn.Module):
    """
    Fully connected deterministic autoencoder for flattened images.
    """

    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 64,
        hidden_dims: tuple[int, int] = (512, 256),
    ) -> None:
        super().__init__()

        h1, h2 = hidden_dims

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes input data into latent representations.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent representations into reconstructed inputs.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs input data.
        """
        z = self.encode(x)
        return self.decode(z)


class BetaVariationalAutoencoder(nn.Module):
    """
    Fully connected beta-Variational Autoencoder for flattened images.
    """

    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 64,
        hidden_dims: tuple[int, int] = (512, 256),
    ) -> None:
        super().__init__()

        h1, h2 = hidden_dims

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input data into mean and log-variance vectors.
        """
        hidden = self.encoder_backbone(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent vectors into reconstructed inputs.
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstructs input data and returns reconstruction, mu and logvar.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar


def get_torch_device(prefer_gpu: bool = True) -> torch.device:
    """
    Returns the available PyTorch device.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_autoencoder_dataloader(
    X: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Creates a DataLoader from a numpy array for autoencoder training.
    """
    _validate_input_array(X, "X")

    if X.min() < 0.0 or X.max() > 1.0:
        raise ValueError("Autoencoder input data must be scaled to [0, 1].")

    X_tensor = torch.from_numpy(X.astype(np.float32))
    dataset = TensorDataset(X_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: torch.device | str | None = None,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Trains a deterministic autoencoder using MSE reconstruction loss.
    """
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if device is None:
        device = get_torch_device()

    device = torch.device(device)
    model = model.to(device)

    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, n_epochs + 1):
        train_loss = _run_autoencoder_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        if val_loader is not None:
            val_loss = _run_autoencoder_epoch(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                train=False,
            )
        else:
            val_loss = np.nan

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

    return history


def vae_loss_function(
    x_reconstructed: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Computes beta-VAE loss components.

    Training objective:
    - reconstruction_sse_per_sample: sum of squared errors per sample, averaged over batch.
    - kl_per_sample: KL divergence per sample, averaged over batch.

    Reporting metric:
    - reconstruction_mse_per_pixel: MSE averaged over batch and pixels.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative.")

    batch_size = x.size(0)

    reconstruction_sse_per_sample = torch.sum(
        (x_reconstructed - x) ** 2
    ) / batch_size

    reconstruction_mse_per_pixel = torch.mean(
        (x_reconstructed - x) ** 2
    )

    kl_per_sample = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp()
    ) / batch_size

    total_loss = reconstruction_sse_per_sample + beta * kl_per_sample

    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_sse_per_sample,
        "reconstruction_mse": reconstruction_mse_per_pixel,
        "kl_loss": kl_per_sample,
    }


def train_beta_vae(
    model: BetaVariationalAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    device: torch.device | str | None = None,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Trains a beta-Variational Autoencoder.
    """
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if beta < 0:
        raise ValueError("beta must be non-negative.")

    if device is None:
        device = get_torch_device()

    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "epoch": [],
        "train_total_loss": [],
        "train_reconstruction_loss": [],
        "train_reconstruction_mse": [],
        "train_kl_loss": [],
        "val_total_loss": [],
        "val_reconstruction_loss": [],
        "val_reconstruction_mse": [],
        "val_kl_loss": [],
    }

    for epoch in range(1, n_epochs + 1):
        train_metrics = _run_beta_vae_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            beta=beta,
            device=device,
            train=True,
        )

        if val_loader is not None:
            val_metrics = _run_beta_vae_epoch(
                model=model,
                data_loader=val_loader,
                optimizer=None,
                beta=beta,
                device=device,
                train=False,
            )
        else:
            val_metrics = {
                "total_loss": np.nan,
                "reconstruction_loss": np.nan,
                "kl_loss": np.nan,
            }
            
        history["epoch"].append(epoch)
        history["train_total_loss"].append(train_metrics["total_loss"])
        history["train_reconstruction_loss"].append(train_metrics["reconstruction_loss"])
        history["train_reconstruction_mse"].append(train_metrics["reconstruction_mse"])
        history["train_kl_loss"].append(train_metrics["kl_loss"])

        history["val_total_loss"].append(val_metrics["total_loss"])
        history["val_reconstruction_loss"].append(val_metrics["reconstruction_loss"])
        history["val_reconstruction_mse"].append(val_metrics["reconstruction_mse"])
        history["val_kl_loss"].append(val_metrics["kl_loss"])

        if verbose:
            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"train_total={train_metrics['total_loss']:.6f} | "
                f"train_rec_sse={train_metrics['reconstruction_loss']:.6f} | "
                f"train_rec_mse={train_metrics['reconstruction_mse']:.6f} | "
                f"train_kl={train_metrics['kl_loss']:.6f} | "
                f"val_total={val_metrics['total_loss']:.6f} | "
                f"val_rec_sse={val_metrics['reconstruction_loss']:.6f} | "
                f"val_rec_mse={val_metrics['reconstruction_mse']:.6f} | "
                f"val_kl={val_metrics['kl_loss']:.6f}"
            )

    return history


def reconstruct_autoencoder(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """
    Reconstructs input samples using a trained deterministic autoencoder.
    """
    _validate_input_array(X, "X")

    if device is None:
        device = get_torch_device()

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    loader = create_autoencoder_dataloader(
        X,
        batch_size=batch_size,
        shuffle=False,
    )

    reconstructed_batches = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_reconstructed = model(batch_x)
            reconstructed_batches.append(batch_reconstructed.cpu().numpy())

    return np.vstack(reconstructed_batches).astype(np.float32)


def reconstruct_beta_vae(
    model: BetaVariationalAutoencoder,
    X: np.ndarray,
    batch_size: int = 256,
    device: torch.device | str | None = None,
    deterministic: bool = True,
) -> np.ndarray:
    """
    Reconstructs input samples using a trained beta-VAE.

    If deterministic=True, reconstructions are decoded from mu.
    """
    _validate_input_array(X, "X")

    if device is None:
        device = get_torch_device()

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    loader = create_autoencoder_dataloader(
        X,
        batch_size=batch_size,
        shuffle=False,
    )

    reconstructed_batches = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            if deterministic:
                mu, _ = model.encode(batch_x)
                batch_reconstructed = model.decode(mu)
            else:
                batch_reconstructed, _, _ = model(batch_x)

            reconstructed_batches.append(batch_reconstructed.cpu().numpy())

    return np.vstack(reconstructed_batches).astype(np.float32)


def encode_autoencoder(
    model: DeterministicAutoencoder,
    X: np.ndarray,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """
    Encodes input samples into deterministic AE latent representations.
    """
    _validate_input_array(X, "X")

    if device is None:
        device = get_torch_device()

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    loader = create_autoencoder_dataloader(
        X,
        batch_size=batch_size,
        shuffle=False,
    )

    latent_batches = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_z = model.encode(batch_x)
            latent_batches.append(batch_z.cpu().numpy())

    return np.vstack(latent_batches).astype(np.float32)


def encode_beta_vae(
    model: BetaVariationalAutoencoder,
    X: np.ndarray,
    batch_size: int = 256,
    device: torch.device | str | None = None,
    return_logvar: bool = False,
):
    """
    Encodes input samples with beta-VAE.

    Returns mu by default. If return_logvar=True, returns (mu, logvar).
    """
    _validate_input_array(X, "X")

    if device is None:
        device = get_torch_device()

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    loader = create_autoencoder_dataloader(
        X,
        batch_size=batch_size,
        shuffle=False,
    )

    mu_batches = []
    logvar_batches = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_mu, batch_logvar = model.encode(batch_x)

            mu_batches.append(batch_mu.cpu().numpy())
            logvar_batches.append(batch_logvar.cpu().numpy())

    mu = np.vstack(mu_batches).astype(np.float32)

    if return_logvar:
        logvar = np.vstack(logvar_batches).astype(np.float32)
        return mu, logvar

    return mu


def decode_autoencoder(
    model: DeterministicAutoencoder,
    Z: np.ndarray,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """
    Decodes deterministic AE latent representations.
    """
    _validate_input_array(Z, "Z")

    if device is None:
        device = get_torch_device()

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    Z_tensor = torch.from_numpy(Z.astype(np.float32))
    loader = DataLoader(
        TensorDataset(Z_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    decoded_batches = []

    with torch.no_grad():
        for (batch_z,) in loader:
            batch_z = batch_z.to(device)
            batch_x = model.decode(batch_z)
            decoded_batches.append(batch_x.cpu().numpy())

    return np.vstack(decoded_batches).astype(np.float32)


def decode_beta_vae(
    model: BetaVariationalAutoencoder,
    Z: np.ndarray,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """
    Decodes beta-VAE latent vectors.
    """
    _validate_input_array(Z, "Z")

    if device is None:
        device = get_torch_device()

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    Z_tensor = torch.from_numpy(Z.astype(np.float32))
    loader = DataLoader(
        TensorDataset(Z_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    decoded_batches = []

    with torch.no_grad():
        for (batch_z,) in loader:
            batch_z = batch_z.to(device)
            batch_x = model.decode(batch_z)
            decoded_batches.append(batch_x.cpu().numpy())

    return np.vstack(decoded_batches).astype(np.float32)


def sample_beta_vae(
    model: BetaVariationalAutoencoder,
    n_samples: int,
    latent_dim: int,
    device: torch.device | str | None = None,
    random_state: int | None = 42,
) -> np.ndarray:
    """
    Samples z ~ N(0, I) and decodes samples with beta-VAE.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")

    if device is None:
        device = get_torch_device()

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    if random_state is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_state)
        z = torch.randn(n_samples, latent_dim, generator=generator).to(device)
    else:
        z = torch.randn(n_samples, latent_dim, device=device)

    with torch.no_grad():
        samples = model.decode(z)

    return samples.cpu().numpy().astype(np.float32)


def save_torch_model(
    model: nn.Module,
    file_path: str | Path,
    config: dict | None = None,
) -> None:
    """
    Saves a PyTorch model checkpoint.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
    }

    torch.save(checkpoint, file_path)


def load_autoencoder_model(
    file_path: str | Path,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, int] = (512, 256),
    device: torch.device | str | None = None,
) -> DeterministicAutoencoder:
    """
    Loads a deterministic autoencoder checkpoint.
    """
    if device is None:
        device = get_torch_device()

    device = torch.device(device)

    model = DeterministicAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )

    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def _run_autoencoder_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> float:
    """
    Runs one deterministic AE training or evaluation epoch.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for (batch_x,) in data_loader:
            batch_x = batch_x.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            batch_reconstructed = model(batch_x)
            loss = criterion(batch_reconstructed, batch_x)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def _run_beta_vae_epoch(
    model: BetaVariationalAutoencoder,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    """
    Runs one beta-VAE training or evaluation epoch.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_samples = 0
    reconstruction_mse_sum = 0.0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for (batch_x,) in data_loader:
            batch_x = batch_x.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            batch_reconstructed, mu, logvar = model(batch_x)

            loss_dict = vae_loss_function(
                x_reconstructed=batch_reconstructed,
                x=batch_x,
                mu=mu,
                logvar=logvar,
                beta=beta,
            )
            batch_size = batch_x.size(0)
            total_loss = loss_dict["total_loss"]

            total_loss_sum += loss_dict["total_loss"].item() * batch_size
            reconstruction_loss_sum += loss_dict["reconstruction_loss"].item() * batch_size
            reconstruction_mse_sum += loss_dict["reconstruction_mse"].item() * batch_size
            kl_loss_sum += loss_dict["kl_loss"].item() * batch_size
            total_samples += batch_size


            if train:
                total_loss.backward()
                optimizer.step()

            batch_size = batch_x.size(0)

            total_loss_sum += loss_dict["total_loss"].item() * batch_size
            reconstruction_loss_sum += loss_dict["reconstruction_loss"].item() * batch_size
            kl_loss_sum += loss_dict["kl_loss"].item() * batch_size
            total_samples += batch_size

    return {
        "total_loss": total_loss_sum / total_samples,
        "reconstruction_loss": reconstruction_loss_sum / total_samples,
        "reconstruction_mse": reconstruction_mse_sum / total_samples,
        "kl_loss": kl_loss_sum / total_samples,
    }


def _validate_input_array(X: np.ndarray, name: str) -> None:
    """
    Validates numpy input arrays.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")

    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape {X.shape}.")

    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinite values.")