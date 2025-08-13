# peft/tuners/our_adapter/layer.py

from __future__ import annotations
import copy
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.func import vmap, functional_call, stack_module_state
from .config import OurAdapterConfig, FuncAdapterConfig
from .discriminator import Discriminator, get_discriminaor_class

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .func_adapter import FuncAdapter

class ConvHelper(nn.Module):
    """Swap dims: (B, T, D) <-> (B, D, T)."""
    def forward(self, x):
        return x.transpose(1, 2)


class OurAdapter(nn.Module):
    def __init__(self, 
                 config: OurAdapterConfig, 
                 num_discriminators: int, 
                 adapter: nn.Module):
        super().__init__()

        diss_cls = get_discriminaor_class(config.discriminator_cfg.type)

        self.discriminators = nn.ModuleList(
            [
                diss_cls(config.discriminator_cfg, config.feature_dim) for _ in range(num_discriminators)
            ]
        )

        if config.add_zero_init_conv_layer:

            conv_layer = nn.Conv1d(
                in_channels=config.out_feature_dim, 
                out_channels=config.out_feature_dim,
                kernel_size=1,
                padding=0
            )

            zero_init_conv_layer = torch.nn.init.zeros_(conv_layer)

            self.adapter = nn.Sequential([
                adapter,
                ConvHelper(),
                zero_init_conv_layer,
                ConvHelper()
            ])
        else:
            self.adapter = adapter



# ---- Layer wrapper: base + adapter ----
class OurAdapterLayer(BaseTunerLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        peft_config: OurAdapterConfig,
        layer_name: str,
        layer_id: int,
        layer_structure: List
    ) -> None:
        self.base_layer = base_layer
        self.peft_config = peft_config
        self.layer_name = layer_name
        self.layer_id = layer_id
        self.layer_structure = layer_structure

        self.adapters = nn.ModuleList([])
        self._discriminators = []
        self._func_adapters = []
        self._mapping = []

        for indice, num_discriminators in enumerate(layer_structure):
            self.add_adapter(num_discriminators)

            for discriminator in self.adapters[-1].discriminators:
                self._discriminators.append(discriminator)
                self._mapping.append(indice)

            self._func_adapters.append(self.adapters[-1].adapter)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_result = self.base_layer(x)
        adapter_result = self._forward_adapter(x)
        return base_result + adapter_result

    def _forward_adapter(self, x: torch.Tensor) -> torch.Tensor:
        losses, state_dicts = self._forward_discriminators_parallelly(x)

        mean_losses = losses.mean(dim=(-2,-1))
        top_1_idx_list = torch.argmin(mean_losses, dim=0).tolist()

        batched_func_adapters = []
        for top_1_idx in top_1_idx_list:
            func_idx = self._mapping[top_1_idx]
            batched_func_adapters.append(self._func_adapters[func_idx])

        params, buffers = stack_module_state(batched_func_adapters)
        prototype = batched_func_adapters[0]

        def forward_adapter(params_i, buffers_i):
            y = functional_call(prototype, (params_i, buffers_i), (x,))
            return y
        
        adapter_result = vmap(forward_adapter)(params, buffers)

        return adapter_result

    def _forward_discriminators_parallelly(self, x: torch.Tensor):
        params, buffers = stack_module_state(self._discriminators)
        prototype = self._discriminators[0]

        def forward_discriminator(params_i, buffers_i):
            loss, state_dict = functional_call(prototype, (params_i, buffers_i), (x,))
            return loss, state_dict

        losses, state_dicts = vmap(forward_discriminator)(params, buffers)
        return losses, state_dicts

    def add_adapter(self, num_discriminators:int = 1):
        if self.peft_config.use_trainable_copy:
            adapter = copy.deepcopy(self.base_layer)
        else:
            adapter = FuncAdapter(
                self.peft_config.func_adapter_cfg, 
                self.peft_config.feature_dim, 
                self.peft_config.out_feature_dim
            )
        self.adapters.append(OurAdapter(self.peft_config, num_discriminators, adapter))

    def add_discriminator(self, adapter_id:int):
        diss_cls = get_discriminaor_class(self.peft_config.discriminator_cfg.type)
        self.adapters[adapter_id].discriminators.append(
            diss_cls(self.peft_config.discriminator_cfg, self.peft_config.feature_dim)
        )
