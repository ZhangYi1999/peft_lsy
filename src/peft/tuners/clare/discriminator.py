import abc
import copy
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import einops
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .config import DiscriminatorConfig


class Discriminator(nn.Module, abc.ABC):
    def __init__(self, config: DiscriminatorConfig, feature_dim: int):
        super().__init__()

        self.config = config
        self.feature_dim: int = feature_dim

        self.feature_fusion: bool = config.feature_fusion

        if self.feature_fusion:
            self.num_tokens: int = config.num_tokens
            self.fused_feature_dim: int = config.fused_feature_dim if config.fused_feature_dim else self.feature_dim

            self.fusion_layer: nn.Linear = nn.Linear(self.num_tokens * self.feature_dim, self.fused_feature_dim)

        self.use_momentum = config.use_momentum
        self.momentum = config.momentum

        self.require_z_score: bool = False
        self.require_update_stats: bool = False

        self.recording_loss = []

        self.info_dict_keys = None

        self.register_buffer('running_mean', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('running_std', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("task_id", torch.tensor(-1, dtype=torch.int64))
        self.register_buffer("connected_adapter_task_id", torch.tensor(-1, dtype=torch.int64))
        self.register_buffer("connected_adapter_indices", torch.tensor(-1, dtype=torch.int64))

    @torch.no_grad
    def update_stats(self, loss: Tensor):
        # loss should be shape (B, 1) or (B,)
        assert loss.ndim <= 2

        max_batches_tracked = torch.tensor(self.config.max_batches_tracked, dtype=torch.int64)
        if self.num_batches_tracked < max_batches_tracked and loss.numel() > 1:
            if self.use_momentum:
                self.running_mean = self.momentum * loss.mean() + (1- self.momentum) * self.running_mean
                self.running_std = self.momentum * loss.std() + (1- self.momentum) * self.running_std
                self.num_batches_tracked += torch.tensor(1, dtype=torch.int64)
            else:
                self.recording_loss.append(loss.mean())
                losses_tensor = torch.stack(self.recording_loss)
                self.running_mean = losses_tensor.mean()

                if self.num_batches_tracked > torch.tensor(0, dtype=torch.int64) and len(self.recording_loss) > 1:
                    self.running_std = losses_tensor.std()

                self.num_batches_tracked += torch.tensor(1, dtype=torch.int64)

    @torch.no_grad
    def compute_z_score(self, mean_loss: Tensor) -> tuple[Tensor, Tensor]:
        z_score = torch.abs((mean_loss - self.running_mean) / self.running_std)
        return z_score

    @abc.abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        raise NotImplementedError

    @abc.abstractmethod
    def unfreeze(self) -> list:
        raise NotImplementedError


# ─── AutoEncoder ──────────────────────────────────────────────────────────────

@DiscriminatorConfig.register_subclass('autoencoder')
@dataclass
class AutoencoderConfig(DiscriminatorConfig):
    hidden_dim: int = None
    latent_dim: int = None    # None → single hidden layer (1-layer); int → two-layer AE with bottleneck
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 8


# Backward-compat alias: 'autoencoder_small' → same config/class
DiscriminatorConfig.register_subclass('autoencoder_small')(AutoencoderConfig)


class AutoEncoder(Discriminator):
    config_class = AutoencoderConfig
    name = "autoencoder"

    def __init__(self, config: AutoencoderConfig, feature_dim: int):
        super().__init__(config, feature_dim)

        input_dim = self.fused_feature_dim if self.feature_fusion else feature_dim
        h = config.hidden_dim
        z = config.latent_dim
        self.use_bottleneck = (z is not None)
        self.use_lora = config.use_lora

        if not self.use_bottleneck:
            # Former AutoEncoderSmall: D → H (ReLU) → D
            self.encoder = nn.Linear(input_dim, h)
            self.decoder = nn.Linear(h, input_dim)
        else:
            if config.use_lora:
                rank = config.lora_rank
                alpha = config.lora_alpha
                self.scaling = alpha / rank

                # Encoder (LoRA style)
                self.encoder_down_A = nn.Linear(input_dim, rank, bias=False)
                self.encoder_down_B = nn.Linear(rank, h, bias=True)
                self.encoder_activation = nn.ReLU()
                self.encoder_up_A = nn.Linear(h, rank, bias=False)
                self.encoder_up_B = nn.Linear(rank, z, bias=True)

                # Decoder (LoRA style)
                self.decoder_down_A = nn.Linear(z, rank, bias=False)
                self.decoder_down_B = nn.Linear(rank, h, bias=True)
                self.decoder_activation = nn.ReLU()
                self.decoder_up_A = nn.Linear(h, rank, bias=False)
                self.decoder_up_B = nn.Linear(rank, input_dim, bias=True)
            else:
                # Normal Encoder: D → H (ReLU) → Z
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, h),
                    nn.ReLU(),
                    nn.Linear(h, z),
                )
                # Normal Decoder: Z → H (ReLU) → D
                self.decoder = nn.Sequential(
                    nn.Linear(z, h),
                    nn.ReLU(),
                    nn.Linear(h, input_dim),
                )

    def encode(self, x: Tensor) -> Tensor:
        if not self.use_bottleneck:
            return torch.relu(self.encoder(x))
        elif self.use_lora:
            x = self.encoder_down_B(self.encoder_down_A(x)) * self.scaling
            x = self.encoder_activation(x)
            x = self.encoder_up_B(self.encoder_up_A(x)) * self.scaling
            return x
        else:
            return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        if self.use_bottleneck and self.use_lora:
            z = self.decoder_down_B(self.decoder_down_A(z)) * self.scaling
            z = self.decoder_activation(z)
            z = self.decoder_up_B(self.decoder_up_A(z)) * self.scaling
            return z
        else:
            return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:

        if self.feature_fusion:
            if x.ndim == 2:
                expanded_feature = x.unsqueeze(-1)  # (B, D) -> (B, 1, D)
            else:
                expanded_feature = x

            if self.config.batch_first:
                flattened_feature = einops.rearrange(expanded_feature, "b t d ... -> b (t d) ...")
            else:
                flattened_feature = einops.rearrange(expanded_feature, "t b d ... -> b (t d) ...")

            input_feature = self.fusion_layer(flattened_feature)
        else:
            input_feature = x

        latent = self.encode(input_feature)
        reconstruction = self.decode(latent)

        reconstruction_loss = F.mse_loss(reconstruction, input_feature, reduction="none")

        info_dict = {
            "reconstruction": reconstruction,
            "loss": reconstruction_loss,
            "latent": latent
        }

        if self.feature_fusion or reconstruction_loss.ndim < 3:
            mean_loss = reconstruction_loss.mean(dim=(-1))  # (B, D) -> (B,)
        else:
            if self.config.batch_first:
                mean_loss = reconstruction_loss.mean(dim=(-2, -1))  # (B, T, D) -> (B,)
            else:
                mean_loss = reconstruction_loss.mean(dim=(-3, -1))  # (T, B, D) -> (B,)

        if self.require_z_score:
            z_score = self.compute_z_score(mean_loss)
            info_dict["z_score"] = z_score

        if self.require_update_stats and self.training:
            self.update_stats(mean_loss)

        info_dict["running_mean"] = self.running_mean
        info_dict["running_std"] = self.running_std
        info_dict["num_batches_tracked"] = self.num_batches_tracked

        self.info_dict_keys = info_dict.keys()

        return mean_loss, info_dict

    def unfreeze(self) -> list:
        training_parameters = []
        for parameter in self.parameters():
            parameter.requires_grad = True
            training_parameters.append(parameter)
        return training_parameters


class BatchedAutoEncoder(nn.Module):
    """Batched parallel inference for AutoEncoder discriminators.

    Stacks weight matrices of N AutoEncoder instances into contiguous tensors
    and runs a single einsum instead of N sequential forward calls.
    Supports both 1-layer (latent_dim=None) and 2-layer (latent_dim=int) architectures.
    LoRA AutoEncoders are not supported (create_batched_discriminator returns None for those).
    """

    def __init__(self, config: AutoencoderConfig, autoencoders: list):
        super().__init__()

        self.config = config
        self.feature_fusion = config.feature_fusion
        self.use_bottleneck = autoencoders[0].use_bottleneck
        # Keep plain list (not nn.ModuleList) — weights are copied to stacked params below
        self._autoencoders = autoencoders

        if self.feature_fusion:
            fus_W, fus_b = [], []
            for ae in autoencoders:
                fus_W.append(copy.deepcopy(ae.fusion_layer.weight.t()))  # (T*D, F) -> (F, T*D) -> t -> (T*D, F)
                fus_b.append(copy.deepcopy(ae.fusion_layer.bias))
            self.fusion_layer_weights = nn.Parameter(
                torch.stack(fus_W, dim=0), requires_grad=False)  # (N, T*D, F)
            self.fusion_layer_bias = nn.Parameter(
                torch.stack(fus_b, dim=0), requires_grad=False)   # (N, F)

        if not self.use_bottleneck:
            # 1-layer: D → H (ReLU) → D
            enc_W, enc_b, dec_W, dec_b = [], [], [], []
            for ae in autoencoders:
                enc_W.append(copy.deepcopy(ae.encoder.weight.t()))  # Linear(D,H): weight(H,D) → t → (D,H)
                enc_b.append(copy.deepcopy(ae.encoder.bias))         # (H,)
                dec_W.append(copy.deepcopy(ae.decoder.weight.t()))  # Linear(H,D): weight(D,H) → t → (H,D)
                dec_b.append(copy.deepcopy(ae.decoder.bias))         # (D,)
            self.encoder_weights = nn.Parameter(
                torch.stack(enc_W, dim=0), requires_grad=False)   # (N, D, H)
            self.encoder_bias = nn.Parameter(
                torch.stack(enc_b, dim=0), requires_grad=False)    # (N, H)
            self.decoder_weights = nn.Parameter(
                torch.stack(dec_W, dim=0), requires_grad=False)   # (N, H, D)
            self.decoder_bias = nn.Parameter(
                torch.stack(dec_b, dim=0), requires_grad=False)    # (N, D)
        else:
            # 2-layer: D → H (ReLU) → Z → H (ReLU) → D
            # encoder: Sequential([Linear(D,H), ReLU, Linear(H,Z)])  indices [0], [2]
            # decoder: Sequential([Linear(Z,H), ReLU, Linear(H,D)])  indices [0], [2]
            enc_W1, enc_b1, enc_W2, enc_b2 = [], [], [], []
            dec_W1, dec_b1, dec_W2, dec_b2 = [], [], [], []
            for ae in autoencoders:
                enc_W1.append(copy.deepcopy(ae.encoder[0].weight.t()))  # (D, H)
                enc_b1.append(copy.deepcopy(ae.encoder[0].bias))         # (H,)
                enc_W2.append(copy.deepcopy(ae.encoder[2].weight.t()))  # (H, Z)
                enc_b2.append(copy.deepcopy(ae.encoder[2].bias))         # (Z,)
                dec_W1.append(copy.deepcopy(ae.decoder[0].weight.t()))  # (Z, H)
                dec_b1.append(copy.deepcopy(ae.decoder[0].bias))         # (H,)
                dec_W2.append(copy.deepcopy(ae.decoder[2].weight.t()))  # (H, D)
                dec_b2.append(copy.deepcopy(ae.decoder[2].bias))         # (D,)
            self.enc_W1 = nn.Parameter(torch.stack(enc_W1, dim=0), requires_grad=False)  # (N, D, H)
            self.enc_b1 = nn.Parameter(torch.stack(enc_b1, dim=0), requires_grad=False)  # (N, H)
            self.enc_W2 = nn.Parameter(torch.stack(enc_W2, dim=0), requires_grad=False)  # (N, H, Z)
            self.enc_b2 = nn.Parameter(torch.stack(enc_b2, dim=0), requires_grad=False)  # (N, Z)
            self.dec_W1 = nn.Parameter(torch.stack(dec_W1, dim=0), requires_grad=False)  # (N, Z, H)
            self.dec_b1 = nn.Parameter(torch.stack(dec_b1, dim=0), requires_grad=False)  # (N, H)
            self.dec_W2 = nn.Parameter(torch.stack(dec_W2, dim=0), requires_grad=False)  # (N, H, D)
            self.dec_b2 = nn.Parameter(torch.stack(dec_b2, dim=0), requires_grad=False)  # (N, D)

    def forward(self, x: Tensor) -> tuple[Tensor, list]:
        # Normalize input to (B, T, D)
        if x.ndim == 2:
            expanded = x.unsqueeze(1)  # (B, D) → (B, 1, D)
        elif not self.config.batch_first:
            expanded = einops.rearrange(x, "t b d ... -> b t d ...")  # (T, B, D) → (B, T, D)
        else:
            expanded = x  # (B, T, D)

        if self.feature_fusion:
            if self.config.batch_first:
                flat = einops.rearrange(expanded, "b t d ... -> b (t d) ...")
            else:
                flat = einops.rearrange(expanded, "t b d ... -> b (t d) ...")
            # Per-discriminator fusion: (N, B, F) → unsqueeze T dim → (N, B, 1, F)
            input_feature = (
                torch.einsum("bd,ndf->nbf", flat, self.fusion_layer_weights)
                + self.fusion_layer_bias[:, None, :]
            ).unsqueeze(2)  # (N, B, 1, F)
            use_n_einsum = True  # input_feature has N dim
        else:
            input_feature = expanded  # (B, T, D)
            use_n_einsum = False

        if not self.use_bottleneck:
            # 1-layer forward
            if use_n_einsum:
                latents = (
                    torch.einsum("nbtd,ndh->nbth", input_feature, self.encoder_weights)
                    + self.encoder_bias[:, None, None, :]
                )
            else:
                latents = (
                    torch.einsum("btd,ndh->nbth", input_feature, self.encoder_weights)
                    + self.encoder_bias[:, None, None, :]
                )
            latents = torch.relu(latents)  # (N, B, T, H)
            reconstructions = (
                torch.einsum("nbth,nhd->nbtd", latents, self.decoder_weights)
                + self.decoder_bias[:, None, None, :]
            )  # (N, B, T, D)
            info_latents = latents
        else:
            # 2-layer forward
            if use_n_einsum:
                l1 = (
                    torch.einsum("nbtd,ndh->nbth", input_feature, self.enc_W1)
                    + self.enc_b1[:, None, None, :]
                )
            else:
                l1 = (
                    torch.einsum("btd,ndh->nbth", input_feature, self.enc_W1)
                    + self.enc_b1[:, None, None, :]
                )
            l1 = torch.relu(l1)  # (N, B, T, H)
            l2 = (
                torch.einsum("nbth,nhz->nbtz", l1, self.enc_W2)
                + self.enc_b2[:, None, None, :]
            )  # (N, B, T, Z)
            r1 = (
                torch.einsum("nbtz,nzh->nbth", l2, self.dec_W1)
                + self.dec_b1[:, None, None, :]
            )
            r1 = torch.relu(r1)  # (N, B, T, H)
            reconstructions = (
                torch.einsum("nbth,nhd->nbtd", r1, self.dec_W2)
                + self.dec_b2[:, None, None, :]
            )  # (N, B, T, D)
            info_latents = l2

        reconstruction_losses = (reconstructions - input_feature).pow(2)  # (N, B, T, D)
        mean_losses = reconstruction_losses.mean(dim=(-2, -1))  # (N, B)

        info_dicts = []
        for i, ae in enumerate(self._autoencoders):
            info_dict = {
                "reconstruction": reconstructions[i],
                "loss": reconstruction_losses[i],
                "latent": info_latents[i],
            }
            if ae.require_z_score:
                info_dict["z_score"] = ae.compute_z_score(mean_losses[i])
            info_dict["running_mean"] = ae.running_mean
            info_dict["running_std"] = ae.running_std
            info_dict["num_batches_tracked"] = ae.num_batches_tracked
            info_dicts.append(info_dict)

        return mean_losses, info_dicts


# ─── VariationalAutoEncoder ───────────────────────────────────────────────────

@DiscriminatorConfig.register_subclass('vae')
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        if self.feature_fusion:
            if x.ndim == 2:
                expanded_feature = x.unsqueeze(-1)  # (B, D) -> (B, 1, D)
            else:
                expanded_feature = x

            if self.config.batch_first:
                flattened_feature = einops.rearrange(expanded_feature, "b t d ... -> b (t d) ...")
            else:
                flattened_feature = einops.rearrange(expanded_feature, "t b d ... -> b (t d) ...")

            input_feature = self.fusion_layer(flattened_feature)
        else:
            input_feature = x

        h = self.encoder(input_feature)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)

        if self.training:
            z = self._reparameterize(mu, logvar)
        else:
            z = mu

        reconstruction = self.decoder(z)

        recon_loss = F.mse_loss(reconstruction, input_feature, reduction="none")

        # KL divergence: sum over latent dim (-1) to handle both (B, Z) and (B, T, Z)
        # Result: (B,) for 2D input, (B, T) for 3D input
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)

        # Broadcast KL to per-feature: unsqueeze(-1) works for both (B,)→(B,1) and (B,T)→(B,T,1)
        kl_expanded = (self.beta * kl_loss).unsqueeze(-1) / input_feature.size(-1)

        total_loss = recon_loss + kl_expanded

        info_dict = {
            "reconstruction": reconstruction,
            "loss": total_loss,
            "mu": mu,
            "logvar": logvar,
            "latent": z,
            "kl_loss": kl_loss
        }

        if self.feature_fusion or total_loss.ndim < 3:
            mean_loss = total_loss.mean(dim=(-1))  # (B, D) -> (B,)
        else:
            if self.config.batch_first:
                mean_loss = total_loss.mean(dim=(-2, -1))  # (B, T, D) -> (B,)
            else:
                mean_loss = total_loss.mean(dim=(-3, -1))  # (T, B, D) -> (B,)

        if self.require_z_score:
            z_score = self.compute_z_score(mean_loss)
            info_dict["z_score"] = z_score

        if self.require_update_stats and self.training:
            self.update_stats(mean_loss)

        info_dict["running_mean"] = self.running_mean
        info_dict["running_std"] = self.running_std
        info_dict["num_batches_tracked"] = self.num_batches_tracked

        self.info_dict_keys = info_dict.keys()

        return mean_loss, info_dict

    def unfreeze(self) -> list:
        training_parameters = []
        for p in self.parameters():
            p.requires_grad = True
            training_parameters.append(p)
        return training_parameters


# ─── RNDDiscriminator ─────────────────────────────────────────────────────────

@DiscriminatorConfig.register_subclass('rnd')
@dataclass
class RNDConfig(DiscriminatorConfig):
    hidden_dim: int = None           # predictor hidden size
    target_hidden_dim: int = None    # target hidden size (defaults to hidden_dim if None)
    embedding_dim: int = None        # output feature size of both nets
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
        self.target.eval()

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
        # Handle (B, T, D) input by reshaping to (B*T, D)
        if x.ndim == 3:
            if self.config.batch_first:
                B, T, D = x.shape
            else:
                T, B, D = x.shape
                x = x.transpose(0, 1)  # → (B, T, D)
                B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
            is_3d = True
        else:
            B = x.shape[0]
            x_flat = x
            is_3d = False

        with torch.no_grad():
            t_feat = self.target(x_flat)   # (B*T, d_emb) or (B, d_emb)

        p_feat = self.predictor(x_flat)    # (B*T, d_emb) or (B, d_emb)

        rnd_err = F.mse_loss(p_feat, t_feat, reduction="none")  # (B*T, d_emb) or (B, d_emb)
        rnd_per_sample = rnd_err.mean(dim=-1)                    # (B*T,) or (B,)

        if is_3d:
            # Reshape back and average over T
            rnd_per_sample = rnd_per_sample.reshape(B, T).mean(dim=1)  # (B,)
            t_feat = t_feat.reshape(B, T, -1)
            p_feat = p_feat.reshape(B, T, -1)

        mean_loss = self.beta * rnd_per_sample  # (B,)

        info_dict = {
            "target_features": t_feat,
            "predictor_features": p_feat,
            "rnd_per_sample": rnd_per_sample,
            "running_mean": self.running_mean,
            "running_std": self.running_std,
            "num_batches_tracked": self.num_batches_tracked,
        }

        if self.require_z_score:
            info_dict["z_score"] = self.compute_z_score(mean_loss)

        if self.require_update_stats and self.training:
            self.update_stats(mean_loss)

        self.info_dict_keys = info_dict.keys()

        return mean_loss, info_dict

    def train(self, mode: bool = True):
        super().train(mode)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self) -> list:
        params = []
        for p in self.predictor.parameters():
            p.requires_grad = True
            params.append(p)
        for p in self.target.parameters():
            p.requires_grad = False
        return params


# ─── LinearMatrixDiscriminator ────────────────────────────────────────────────

@DiscriminatorConfig.register_subclass('linear_matrix')
@dataclass
class LinearMatrixConfig(DiscriminatorConfig):
    l2_weight: float = 1e-3


class LinearMatrixDiscriminator(Discriminator):
    """Linear discriminator: score = x @ w; loss = -(x @ w) + l2_weight * ||w||^2.

    With L2 regularization, at convergence w ∝ E[x] (task mean direction).
    The L2 term is a scalar constant per forward call and does not affect argmin routing.
    """
    config_class = LinearMatrixConfig
    name = "linear_matrix"

    def __init__(self, config: LinearMatrixConfig, feature_dim: int):
        super().__init__(config, feature_dim)
        input_dim = self.fused_feature_dim if self.feature_fusion else feature_dim
        self.w = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.l2_weight = config.l2_weight

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        if self.feature_fusion:
            if x.ndim == 2:
                expanded_feature = x.unsqueeze(-1)
            else:
                expanded_feature = x

            if self.config.batch_first:
                flattened_feature = einops.rearrange(expanded_feature, "b t d ... -> b (t d) ...")
            else:
                flattened_feature = einops.rearrange(expanded_feature, "t b d ... -> b (t d) ...")

            input_feature = self.fusion_layer(flattened_feature)
        else:
            input_feature = x

        if input_feature.ndim == 2:
            score = input_feature @ self.w  # (B,)
        else:
            # (B, T, D) batch_first or (T, B, D)
            if self.config.batch_first:
                score = torch.einsum("btd,d->bt", input_feature, self.w).mean(1)  # (B,)
            else:
                score = torch.einsum("tbd,d->tb", input_feature, self.w).mean(0)  # (B,)

        l2_reg = self.l2_weight * self.w.pow(2).sum()
        mean_loss = -score + l2_reg  # (B,); l2 is scalar, broadcasts

        info_dict = {
            "score": score,
            "running_mean": self.running_mean,
            "running_std": self.running_std,
            "num_batches_tracked": self.num_batches_tracked,
        }

        if self.require_z_score:
            info_dict["z_score"] = self.compute_z_score(mean_loss)

        if self.require_update_stats and self.training:
            self.update_stats(mean_loss)

        self.info_dict_keys = info_dict.keys()

        return mean_loss, info_dict

    def unfreeze(self) -> list:
        self.w.requires_grad = True
        return [self.w]


class BatchedLinearMatrix(nn.Module):
    """Batched parallel inference for LinearMatrixDiscriminator.

    Stacks w vectors into W_stacked (N, D) and computes all N scores in one einsum.
    """

    def __init__(self, config: LinearMatrixConfig, discriminators: list):
        super().__init__()
        self.config = config
        self.feature_fusion = config.feature_fusion
        self._discriminators = discriminators

        W = torch.stack([d.w.data for d in discriminators], dim=0)  # (N, D)
        self.W_stacked = nn.Parameter(W, requires_grad=False)

        if self.feature_fusion:
            fus_W, fus_b = [], []
            for d in discriminators:
                fus_W.append(copy.deepcopy(d.fusion_layer.weight.t()))
                fus_b.append(copy.deepcopy(d.fusion_layer.bias))
            self.fusion_layer_weights = nn.Parameter(
                torch.stack(fus_W, dim=0), requires_grad=False)  # (N, T*D, F)
            self.fusion_layer_bias = nn.Parameter(
                torch.stack(fus_b, dim=0), requires_grad=False)   # (N, F)

    def forward(self, x: Tensor) -> tuple[Tensor, list]:
        if self.feature_fusion:
            if x.ndim == 2:
                expanded = x.unsqueeze(1)
            elif not self.config.batch_first:
                expanded = einops.rearrange(x, "t b d ... -> b t d ...")
            else:
                expanded = x

            if self.config.batch_first:
                flat = einops.rearrange(expanded, "b t d ... -> b (t d) ...")
            else:
                flat = einops.rearrange(expanded, "t b d ... -> b (t d) ...")

            # Per-discriminator fusion: (N, B, F)
            input_feature = (
                torch.einsum("bd,ndf->nbf", flat, self.fusion_layer_weights)
                + self.fusion_layer_bias[:, None, :]
            )
            # scores: (N, B)
            scores = torch.einsum("nbf,nf->nb", input_feature, self.W_stacked)
        else:
            if x.ndim == 2:
                scores = torch.einsum("bd,nd->nb", x, self.W_stacked)  # (N, B)
            elif self.config.batch_first:
                scores = torch.einsum("btd,nd->nbt", x, self.W_stacked).mean(2)  # (N, B)
            else:
                scores = torch.einsum("tbd,nd->nbt", x, self.W_stacked).mean(2)  # (N, B)

        losses = -scores  # (N, B); no L2 at eval — correct for argmin

        info_dicts = []
        for i, d in enumerate(self._discriminators):
            info_dicts.append({
                "score": scores[i],
                "running_mean": d.running_mean,
                "running_std": d.running_std,
                "num_batches_tracked": d.num_batches_tracked,
            })

        return losses, info_dicts


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_batched_discriminator(discriminators) -> Optional[nn.Module]:
    """Return a batched discriminator if all discriminators are the same type and support batching.

    Returns None if types are mixed or the discriminator type has no batched implementation
    (e.g. VAE, RND, LoRA AutoEncoder) — callers should fall back to sequential execution.
    """
    if not discriminators:
        return None
    disc_list = list(discriminators)
    if not all(type(d) is type(disc_list[0]) for d in disc_list):
        return None  # mixed types: sequential fallback

    first = disc_list[0]
    if isinstance(first, AutoEncoder):
        if first.use_lora:
            return None  # LoRA weights are not batchable
        return BatchedAutoEncoder(first.config, disc_list)
    if isinstance(first, LinearMatrixDiscriminator):
        return BatchedLinearMatrix(first.config, disc_list)
    return None  # VAE, RND, unknown: no batched version


# ─── Registry helper ─────────────────────────────────────────────────────────

def get_discriminaor_class(name: str) -> type:
    """Get the discriminator class given a registered name."""
    _map = {
        "autoencoder": AutoEncoder,
        "autoencoder_small": AutoEncoder,  # alias: same class, latent_dim=None
        "vae": VariationalAutoEncoder,
        "rnd": RNDDiscriminator,
        "linear_matrix": LinearMatrixDiscriminator,
    }
    if name not in _map:
        raise NotImplementedError(f"Discriminator with name '{name}' is not implemented.")
    return _map[name]
