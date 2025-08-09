# peft/tuners/our_adapter/layer.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn

from .config import OurAdapterConfig, FuncAdapterConfig
from .discriminator import Discriminator

# ---- Func adapter ----
class FuncAdapter(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.down_proj = nn.Linear(in_features, hidden_dim, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.up_proj = nn.Linear(hidden_dim, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.dropout(self.activation(self.down_proj(x))))



# ---- Block + adapter container ----
class Block(nn.Module):
    def __init__(self, config: FuncAdapterConfig, task_id: int) -> None:
        super().__init__()
        self.task_id = task_id
        self.num_discriminators = 0

        self.discriminators = nn.ModuleList([])
        self.adapter = FuncAdapter(config)
        

class OurAdapter(nn.Module):
    """
    Manages multiple blocks. Selection mode when active_task == -1, otherwise a specific disc.
    Stores metrics for diagnostics.
    """
    def __init__(
        self,
        config: OurAdapterConfig,
        in_features: int,
        out_features: int,
        disc_ctor: Any = None,
        disc_cfg: Any = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = config.r
        self.dropout_p = config.dropout
        self.blocks = nn.ModuleList([
            Block(disc_ctor, disc_cfg, in_features, self.hidden_dim, out_features, 1, self.dropout_p,
                  func_ctor=config.func_adapter_class)
        ])
        # -1 means selection mode
        self.active_task: int = -1
        self.forward_block: int = -1
        self.forward_disc: int = -1
        self.metrics: Dict[str, Any] = {
            "loss": None,
            "aux": None,
            "chosen_block": None,
            "chosen_disc": None,
        }

    def add_adapters(self, num_blocks: int) -> None:
        for _ in range(num_blocks):
            self.blocks.append(
                Block(
                    self.config.discriminator_class,
                    None,
                    self.in_features,
                    self.hidden_dim,
                    self.out_features,
                    1,
                    self.dropout_p,
                    func_ctor=self.config.func_adapter_class
                )
            )

    def add_discriminators(self, block_id: int, num_discriminators: int) -> None:
        assert 0 <= block_id < len(self.blocks)
        self.blocks[block_id].add_discriminators(self.config.discriminator_class, None, num_discriminators)

    def set_active_task(self, task_idx: int) -> None:
        total = sum(len(b.discriminators) for b in self.blocks)
        assert 0 <= task_idx < total
        self.active_task = task_idx
        count = 0
        for b_idx, block in enumerate(self.blocks):
            n = len(block.discriminators)
            if task_idx < count + n:
                self.forward_block, self.forward_disc = b_idx, task_idx - count
                return
            count += n

    def set_forward_all(self) -> None:
        self.active_task = -1
        self.forward_block = -1
        self.forward_disc = -1

    def set_forward(self, block_id: int, disc_id: int) -> None:
        assert 0 <= block_id < len(self.blocks)
        assert 0 <= disc_id < len(self.blocks[block_id].discriminators)
        self.forward_block, self.forward_disc = block_id, disc_id

    def trainable_parameters(self) -> List[nn.Parameter]:
        if self.active_task == -1:
            return []
        b, d = self.forward_block, self.forward_disc
        disc = self.blocks[b].discriminators[d]
        func = self.blocks[b].func
        return list(disc.parameters()) + list(func.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Selection mode
        if self.active_task == -1:
            all_losses: List[torch.Tensor] = []
            all_aux: List[Dict[str, torch.Tensor]] = []
            index_map: List[int] = []
            flat_pairs: List[Tuple[int, int]] = []
            for b_idx, block in enumerate(self.blocks):
                for d_idx in range(len(block.discriminators)):
                    flat_pairs.append((b_idx, d_idx))
                    index_map.append(b_idx)
            with torch.no_grad():
                for b_idx, d_idx in flat_pairs:
                    loss, aux = self.blocks[b_idx].discriminators[d_idx](x)
                    if torch.is_tensor(loss) and loss.ndim > 0:
                        while loss.ndim > 0:
                            loss = loss.mean(dim=-1)
                    elif not torch.is_tensor(loss):
                        loss = x.new_tensor(float(loss))
                    all_losses.append(loss)
                    all_aux.append(aux)
            losses_tensor = torch.stack(all_losses)
            top1 = int(torch.argmin(losses_tensor).item())
            chosen_block = index_map[top1]
            offset = 0
            for b_idx, block in enumerate(self.blocks):
                n = len(block.discriminators)
                if top1 < offset + n:
                    chosen_disc = top1 - offset
                    break
                offset += n
            else:
                chosen_disc = 0
            self.metrics = {
                "loss": losses_tensor.detach(),
                "aux": all_aux,
                "chosen_block": chosen_block,
                "chosen_disc": chosen_disc,
            }
            with torch.no_grad():
                return self.blocks[chosen_block].func(x)

        # Task mode
        b, d = self.forward_block, self.forward_disc
        # compute active pair from active_task
        offset = 0
        for b_idx, block in enumerate(self.blocks):
            n = len(block.discriminators)
            if self.active_task < offset + n:
                active_b, active_d = b_idx, self.active_task - offset
                break
            offset += n
        else:
            active_b, active_d = 0, 0

        disc = self.blocks[b].discriminators[d]
        func = self.blocks[b].func
        if (b, d) == (active_b, active_d):
            loss, aux = disc(x)
            out = func(x)
        else:
            with torch.no_grad():
                loss, aux = disc(x)
                out = func(x)
        self.metrics = {
            "loss": loss,
            "aux": aux,
            "chosen_block": b,
            "chosen_disc": d,
        }
        return out

# ---- Layer wrapper: base + adapter ----
class OurAdapterLayer:
    def __init__(
        self,
        model: Any,
        target: nn.Module,
        target_forward: Any,
        layer_number: int,
        config: OurAdapterConfig,
    ) -> None:
        self.model = model
        self.target = target
        self.target_forward = target_forward
        self.layer_number = layer_number
        self.config = config
        in_features = getattr(target, "in_features", None) or 1
        out_features = getattr(target, "out_features", None) or 1
        self.adapter = OurAdapter(config, int(in_features), int(out_features))
        if hasattr(model, "_our_adapter_layers"):
            model._our_adapter_layers.append(self)
        else:
            model._our_adapter_layers = [self]

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        y = self.target_forward(x, *args, **kwargs)
        # allow global disable (e.g. for saving base)
        if getattr(self.model, "disabled", False):
            return y
        delta = self.adapter(x)
        if delta.dtype != y.dtype:
            delta = delta.to(y.dtype)
        return y + delta
