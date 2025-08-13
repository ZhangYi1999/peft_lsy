import abc
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import numpy as np
from dataclasses import dataclass

from .config import DiscriminatorConfig


class Discriminator(nn.Module, abc.ABC):
    def __init__(self, config:DiscriminatorConfig, feature_dim: int):
        super().__init__()

        self.config = config
        self.feature_dim = feature_dim

        self.momentum = 0.1
        self.use_momentum = True

        self.recording_loss = []

        self.register_buffer('running_mean', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('running_std', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.int64))

    def update_stats(self, loss: Tensor):
        # loss should be shape (B, 1) or (B,)
        assert len(loss.shape) <= 2

        if self.use_momentum:
            if self.num_batches_tracked < self.config.max_batches_tracked:
                self.running_mean = self.momentum * loss.mean() + (1- self.momentum) * self.running_mean
                self.running_std = self.momentum * loss.std() + (1- self.momentum) * self.running_std
                self.num_batches_tracked += torch.tensor(1, dtype=torch.int64)
        else:
            if self.num_batches_tracked < self.config.max_batches_tracked:
                self.recording_loss.append(loss.mean())
                losses_tensor = torch.stack(self.recording_loss)
                self.running_mean = losses_tensor.mean()

                if self.num_batches_tracked > 0 and len(self.recording_loss) > 1:
                    self.running_std = losses_tensor.std()

                self.num_batches_tracked += torch.tensor(1, dtype=torch.int64)

    @torch.no_grad
    def compute_z_score(self, x: Tensor) -> tuple[Tensor, Tensor]:
        loss, _ = self.forward(x)
        mean_loss = loss.mean()
        z_score = torch.abs((mean_loss - self.running_mean) / self.running_std)
        return z_score, mean_loss
    
    @abc.abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def unfreeze(self) -> list:
        raise NotImplementedError


@DiscriminatorConfig.register_subclass("autoencoder")
@dataclass
class AutoencoderConfig(DiscriminatorConfig):
    hidden_dim: int = None
    latent_dim: int = None


class AutoEncoder(Discriminator):
    config_class = AutoencoderConfig
    name = "autoencoder"

    def __init__(self, config: AutoencoderConfig, feature_dim: int):
        super().__init__(config, feature_dim)

        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        
        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, feature_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        reconstruction_loss = F.mse_loss(reconstruction, x, reduction="none")

        return reconstruction_loss, {"reconstruction": reconstruction}
    
    def unfreeze(self) -> list:
        training_parameters = []
        for parameter in self.parameters():
            parameter.requires_grad = True
            training_parameters.append(parameter)

        return training_parameters


@DiscriminatorConfig.register_subclass("vae")
@dataclass
class VAEConfig(DiscriminatorConfig):
    hidden_dim: int = None
    latent_dim: int = None
    beta: float = 1.0  # KL weight


class VariationalAutoEncoder(Discriminator):
    config_class = VAEConfig
    name = "vae"

    def __init__(self, config: VAEConfig, feature_dim: int):
        super().__init__(config, feature_dim)
        h = config.hidden_dim
        z = config.latent_dim

        # Encoder: x -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, h),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(h, z)
        self.enc_logvar = nn.Linear(h, z)

        # Decoder: z -> hidden -> x_hat
        self.decoder = nn.Sequential(
            nn.Linear(z, h),
            nn.ReLU(),
            nn.Linear(h, feature_dim),
        )

        self.beta = config.beta
        self.feature_dim = feature_dim
        self.latent_dim = z

    @staticmethod
    def _reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        # std = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        h = self.encoder(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)

        # Reparameterization trick
        if self.training:
            z = self._reparameterize(mu, logvar)
        else:
            # use mean at eval for deterministic recon
            z = mu

        reconstruction = self.decoder(z)

        # Per-element reconstruction loss (same as AE)
        recon_loss = F.mse_loss(reconstruction, x, reduction="none")

        # KL divergence per sample: shape [batch]
        # KL(N(mu, sigma) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)

        # Make KL term broadcastable to per-feature loss: distribute evenly across features
        kl_expanded = (self.beta * kl_loss).unsqueeze(1) / x.size(1)

        total_loss = recon_loss + kl_expanded

        state_dict = {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "latent": z,
            "kl_loss": kl_loss,
        }
        return total_loss, state_dict

    def unfreeze(self) -> list:
        training_parameters = []
        for p in self.parameters():
            p.requires_grad = True
            training_parameters.append(p)
        return training_parameters


@DiscriminatorConfig.register_subclass("rnd")
@dataclass
class RNDConfig(DiscriminatorConfig):
    # MLP sizes
    hidden_dim: int = None          # predictor hidden size
    target_hidden_dim: int = None   # target hidden size (defaults to hidden_dim if None)
    embedding_dim: int = None       # output feature size of both nets

    # Scale to weight the RND term relative to other discriminators
    beta: float = 1.0


class RNDDiscriminator(Discriminator):
    config_class = RNDConfig
    name = "rnd"

    def __init__(self, config: RNDConfig, feature_dim: int):
        super().__init__(config, feature_dim)

        h_pred = config.hidden_dim
        h_tgt = config.target_hidden_dim or h_pred
        d_emb = config.embedding_dim

        # Target network: randomly initialized, never trained
        self.target = nn.Sequential(
            nn.Linear(feature_dim, h_tgt),
            nn.ReLU(),
            nn.Linear(h_tgt, d_emb),
        )
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()  # remain in eval; no dropout/bn anyway but explicit

        # Predictor network: trained to match target features
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, h_pred),
            nn.ReLU(),
            nn.Linear(h_pred, d_emb),
        )

        self.beta = config.beta
        self.feature_dim = feature_dim
        self.embedding_dim = d_emb

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        with torch.no_grad():
            t_feat = self.target(x)

        p_feat = self.predictor(x)

        # RND error per element in embedding space [B, d_emb]
        rnd_err = F.mse_loss(p_feat, t_feat, reduction="none")  # [B, d_emb]

        # Reduce to per-sample scalar [B] (mean over embedding)
        rnd_per_sample = rnd_err.mean(dim=1)

        # Broadcast to per-feature like AE: [B, F]
        rnd_per_feature = (self.beta * rnd_per_sample).unsqueeze(1) / x.size(1)

        state_dict = {
            "target_features": t_feat,
            "predictor_features": p_feat,
            "rnd_per_sample": rnd_per_sample,
        }
        return rnd_per_feature, state_dict

    def train(self, mode: bool = True):
        # keep target frozen & in eval regardless of outer mode
        super().train(mode)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self) -> list:
        # Only predictor should train
        params = []
        for p in self.predictor.parameters():
            p.requires_grad = True
            params.append(p)
        # ensure target stays frozen
        for p in self.target.parameters():
            p.requires_grad = False
        return params


def get_discriminaor_class(name: str) -> Discriminator:
    """Get the discriminaor's class and config class given a name (matching the discriminaor class' `name` attribute)."""
    if name == "autoencoder":
        return AutoEncoder
    elif name == "vae":
        return VariationalAutoEncoder
    else:
        raise NotImplementedError(f"Discriminator with name {name} is not implemented.")

