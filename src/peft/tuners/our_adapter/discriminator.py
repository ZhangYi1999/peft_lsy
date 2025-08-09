import abc
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import numpy as np
from dataclasses import dataclass

from .config import DiscriminatorConfig


class Discriminator(nn.Module, abc.ABC):
    def __init__(self, config:DiscriminatorConfig):
        super().__init__()

        self.config = config

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
    feature_dim: int = None
    max_batches_tracked: int = 2000

    hidden_dim: int = None
    latent_dim: int = None


class AutoEncoder(Discriminator):
    config_class = AutoencoderConfig
    name = "autoencoder"

    def __init__(self, config: AutoencoderConfig):
        super().__init__(config)

        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        
        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.feature_dim),
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



def get_discriminaor_class(name: str) -> Discriminator:
    """Get the discriminaor's class and config class given a name (matching the discriminaor class' `name` attribute)."""
    if name == "autoencoder":
        return AutoEncoder
    else:
        raise NotImplementedError(f"Discriminator with name {name} is not implemented.")

