# peft/tuners/our_adapter/layer.py

from __future__ import annotations
import copy
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.func import vmap, functional_call, stack_module_state
import einops
from .config import OurAdapterConfig, FuncAdapterConfig
from .discriminator import Discriminator, get_discriminaor_class

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .func_adapter import FuncAdapter

class ConvHelper(nn.Module):
    """Swap dims: (B, T, D) <-> (B, D, T)."""
    def forward(self, x):
        return x.transpose(1, 2)


class FuncAdapterWrapper(nn.Module):
    def __init__(self, 
                 config: OurAdapterConfig, 
                 adapter: nn.Module):
        super().__init__()

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

    def forward(self, x):
        return self.adapter(x)



# ---- Layer wrapper: base + adapter ----
class OurAdapterLayer(nn.Module, BaseTunerLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        peft_config: OurAdapterConfig,
        adapter_name: str,
        layer_name: str,
        layer_id: int,
        base_layer_name: str,
        num_adapters: int,
        num_discriminators: int
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.layer_name = layer_name
        self.layer_id = layer_id
        self.base_layer_name = base_layer_name
        self.num_adapters = num_adapters
        self.num_discriminators = num_discriminators

        # create adapters
        self.our_adapter_func_adapters: nn.ModuleList[FuncAdapterWrapper] = \
            nn.ModuleList([self._create_adapter() for _ in range(num_adapters)])

        # create discriminators
        self.our_adapter_discriminators: nn.ModuleList[Discriminator] = \
            nn.ModuleList([self._create_discriminator() for _ in range(num_discriminators)])

        self._state_dicts: dict = {}
        self._active_task: int = -1
        self._forwarded_adapter_id: int = -1
        self._forwarded_discriminator_id: int = -1
        self._train_discriminator: bool = False
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_result = self.base_layer(x)
        adapter_result = self._forward_adapter(x)
        result = base_result + adapter_result
        return result

    def _forward_adapter(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # during training

            # forward specific discriminator
            if self._train_discriminator:
                _, state_dict = self.our_adapter_discriminators[self._forwarded_discriminator_id](x)

                if self._forwarded_discriminator_id == -1:
                    discriminator_id = len(self.our_adapter_discriminators) - 1  
                else:
                    discriminator_id = self._forwarded_discriminator_id
                self._state_dicts[f"discriminator_{discriminator_id}"] = state_dict

            # forward specific adapter
            adapter_result = self.our_adapter_func_adapters[self._forwarded_adapter_id](x)
        else:
            # during evaluation

            losses, state_dicts = self._forward_discriminators_parallelly(x)

            for indice, state_dict in enumerate(state_dicts):
                self._state_dicts[f"discriminator_{indice}"] = state_dict

            batched_func_adapters = []

            top_1_idx_list = torch.argmin(losses, dim=0).tolist()

            self._state_dicts["top_1_idx_list"] = top_1_idx_list

            for top_1_idx in top_1_idx_list:
                func_idx = self.our_adapter_discriminators[top_1_idx].connected_adapter_indices
                batched_func_adapters.append(self.our_adapter_func_adapters[func_idx])

            params, buffers = stack_module_state(batched_func_adapters)
            prototype = batched_func_adapters[0]

            def forward_one_func_adapter(params_i, buffers_i, input_i):
                y = functional_call(prototype, (params_i, buffers_i), (input_i,))
                return y
            
            if x.ndim == 3 and x.shape[0] == self.peft_config.num_tokens:
                batch_first = False
                adapter_input = einops.rearrange(x, "t b d ... -> b t d ... ")
            else:
                batch_first = True
                adapter_input = x
            
            adapter_result = vmap(forward_one_func_adapter)(params, buffers, adapter_input)

            if not batch_first:
                adapter_result = einops.rearrange(adapter_result, "b t d ... -> t b d ... ")

        return adapter_result

    def _forward_discriminators_parallelly(self, x: torch.Tensor):
        params, buffers = stack_module_state(self.our_adapter_discriminators)
        prototype = self.our_adapter_discriminators[0]

        def forward_one_discriminator(params_i, buffers_i):
            loss, state_dict = functional_call(prototype, (params_i, buffers_i), (x,))
            return loss, state_dict

        losses, state_dicts = vmap(forward_one_discriminator)(params, buffers)
        return losses, state_dicts

    def _create_adapter(self):
        if self.peft_config.use_trainable_copy:
            adapter = copy.deepcopy(self.base_layer)
        else:
            adapter = FuncAdapter(
                self.peft_config.func_adapter_cfg, 
                self.peft_config.feature_dim, 
                self.peft_config.out_feature_dim
            )
        return FuncAdapterWrapper(self.peft_config, adapter)
    
    def _create_discriminator(self):
        disc_cls = get_discriminaor_class(self.peft_config.discriminator_cfg.type)
        return disc_cls(self.peft_config.discriminator_cfg, self.peft_config.feature_dim)

    def add_adapter_and_discriminator(self, new_task_id:int):
        new_adapter = self._create_adapter()
        new_adapter.task_id = torch.tensor(new_task_id, dtype=torch.int64)
        new_adapter.to(next(self.base_layer.parameters()).device)
        self.our_adapter_func_adapters.append(new_adapter)
        self.num_adapters += 1

        adapter_parameter = []
        for parameter in new_adapter.parameters():
            parameter.requires_grad = True
            adapter_parameter.append(parameter)
        
        discriminator_parameter = self.add_discriminator(new_task_id, new_task_id)

        return adapter_parameter, discriminator_parameter

    def add_discriminator(self, connected_adapter_indices:int, new_task_id:int):
        new_discriminator = self._create_discriminator()
        new_discriminator.task_id = torch.tensor(new_task_id, dtype=torch.int64)
        new_discriminator.connected_adapter_indices = torch.tensor(connected_adapter_indices, dtype=torch.int64)
        new_discriminator.connected_adapter_task_id = self.our_adapter_func_adapters[connected_adapter_indices].task_id
        new_discriminator.to(next(self.base_layer.parameters()).device)
        self.our_adapter_discriminators.append(new_discriminator)
        self.num_discriminators += 1

        discriminator_parameter = []
        for parameter in new_discriminator.parameters():
            parameter.requires_grad = True
            discriminator_parameter.append(parameter)

        return discriminator_parameter
    
    def train_discriminator(self, train_discriminator:bool):
        self._train_discriminator = train_discriminator

    def track_z_score(self, require_z_score:bool):
        for discriminator in self.our_adapter_discriminators:
            discriminator.require_z_score = require_z_score

    def update_stats(self, require_update_stats:bool):
        for discriminator in self.our_adapter_discriminators:
            discriminator.require_update_stats = require_update_stats

    def get_adapter_id_by_discriminator_id(self, discriminator_id):
        return self.our_adapter_discriminators[discriminator_id].connected_adapter_indices.item()
    
    @property
    def state_dicts(self):
        return self._state_dicts