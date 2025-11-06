# peft/tuners/our_adapter/layer.py

from __future__ import annotations
import copy
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.func import vmap, functional_call, stack_module_state
import einops
from .config import CLAREConfig, FuncAdapterConfig, CLAREModuleConfig
from .discriminator import Discriminator, get_discriminaor_class
from .base_wrapper import LoRABaseWrapper
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .func_adapter import FuncAdapter

STACK_FORWARD = False

class ConvHelper(nn.Module):
    """Swap dims: (B, T, D) <-> (B, D, T)."""
    def forward(self, x):
        return x.transpose(1, 2)


class FuncAdapterWrapper(nn.Module):
    def __init__(self, 
                 config: CLAREConfig, 
                 adapter: nn.Module):
        super().__init__()

        self.add_zero_init_conv_layer = config.add_zero_init_conv_layer
        self.func_adapter = None  # Will be set below
        self.lora_adapter = {}  # Store LoRA parameters for PEFT state management

        if config.add_zero_init_conv_layer:

            conv_layer = nn.Conv1d(
                in_channels=config.out_feature_dim, 
                out_channels=config.out_feature_dim,
                kernel_size=1,
                padding=0
            )

            # Initialize weights and bias to zero
            nn.init.constant_(conv_layer.weight, 0.0)
            if conv_layer.bias is not None:
                nn.init.constant_(conv_layer.bias, 0.0)

            self.func_adapter = nn.Sequential(
                adapter,
                ConvHelper(),
                conv_layer,
                ConvHelper()
            )
        else:
            self.func_adapter = adapter

    def forward(self, x):
        if x.ndim == 2 and self.add_zero_init_conv_layer:
            x = x.squeeze(0)
            y = self.func_adapter(x)
            y = y.unsqueeze(0)
            return y
        else:
            return self.func_adapter(x)
            
    def _save_to_state_dict(self, destination, prefix, keep_vars: bool):
        """Handle LoRA parameter saving for PEFT compatibility"""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Save LoRA parameters if they exist
        if self.lora_adapter:
            for name, adapter_dict in self.lora_adapter.items():
                for param_name, param in adapter_dict.items():
                    key = prefix + f'lora_adapter.{name}.{param_name}'
                    destination[key] = param if keep_vars else param.detach()
                    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Handle LoRA parameter loading for PEFT compatibility"""
        # Extract and remove LoRA parameters from state dict
        lora_params = {}
        keys_to_remove = []
        
        for key in state_dict:
            if key.startswith(prefix + 'lora_adapter.'):
                # Parse the key to get module name and param type
                parts = key[len(prefix + 'lora_adapter.'):].split('.')
                module_name = parts[0]
                param_name = parts[1]
                
                if module_name not in lora_params:
                    lora_params[module_name] = {}
                lora_params[module_name][param_name] = state_dict[key]
                keys_to_remove.append(key)
                
        # Remove LoRA keys from state dict
        for key in keys_to_remove:
            del state_dict[key]
            
        # Load regular parameters
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        
        # Store LoRA parameters
        self.lora_adapter = lora_params



# ---- Layer wrapper: base + adapter ----
class CLARELayer(nn.Module, BaseTunerLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        peft_config: CLAREConfig,
        module_config: CLAREModuleConfig,
        adapter_name: str,
        layer_name: str,
        layer_id: int,
        base_layer_name: str,
        num_adapters: int,
        num_discriminators: int
    ) -> None:
        super().__init__()
        self.peft_config = peft_config
        self.module_config = module_config
        self.adapter_name = adapter_name
        self.layer_name = layer_name
        self.layer_id = layer_id
        self.base_layer_name = base_layer_name
        self.num_adapters = num_adapters
        self.num_discriminators = num_discriminators
        self.use_lora = self.module_config.func_adapter_cfg.use_lora

        # Initialize adapters structure first, regardless of LoRA or not
        new_func_adapters_list = nn.ModuleList([self._create_adapter() for _ in range(num_adapters)])
        self.our_adapter_func_adapters: nn.ModuleDict[str, nn.ModuleList[FuncAdapterWrapper]] = \
            nn.ModuleDict({self.adapter_name:new_func_adapters_list})

        # Initialize base layer depending on whether using LoRA or not
        if self.use_lora:
            
            self.base_layer = LoRABaseWrapper(base_layer)
            # Initialize LoRA modules using the LoRA parameters from func_adapter_cfg
            self.base_layer.wrap_linear_layers(self.module_config.func_adapter_cfg)
            # Add adapters for each task and store them in our_adapter_func_adapters
            for i in range(num_adapters):
                # Create LoRA adapter in base layer
                self.base_layer.add_adapter(i)
                # Store reference to all LoRA modules for this adapter
                lora_adapter = {}
                for name, lora_module in self.base_layer.lora_modules.items():
                    lora_adapter[name] = {
                        'lora_A': lora_module.lora_a[str(i)],
                        'lora_B': lora_module.lora_b[str(i)]
                    }
                # Store in our standard adapter structure
                self.our_adapter_func_adapters[self.adapter_name][i].lora_adapter = lora_adapter
        else:
            self.base_layer = base_layer

        self._base_layer_device = next(self.base_layer.parameters()).device
        self._base_layer_dtype = next(self.base_layer.parameters()).dtype

        # create discriminators
        new_discriminators_list = nn.ModuleList([self._create_discriminator() for _ in range(num_discriminators)])
        self.our_adapter_discriminators: nn.ModuleDict[str, nn.ModuleList[Discriminator]] = \
            nn.ModuleDict({self.adapter_name:new_discriminators_list})

        self._info_dicts: dict = {}
        self._active_task: int = -1
        self._forwarded_adapter_id: int = -1
        self._forwarded_discriminator_id: int = -1
        self._train_discriminator: bool = False
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_lora:
            # LoRA case, but still use discriminators to choose adapter
            result = self._forward_lora_adapter(x)
        else:
            # Original case with separate adapters
            base_result = self.base_layer(x)
            adapter_result = self._forward_adapter(x)
            result = base_result + adapter_result
        return result

    def _forward_lora_adapter(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # during training
            # forward specific discriminator
            if self._train_discriminator:
                _, info_dict = self.our_adapter_discriminators[self.adapter_name][self._forwarded_discriminator_id](x)

                if self._forwarded_discriminator_id == -1:
                    discriminator_id = len(self.our_adapter_discriminators[self.adapter_name]) - 1  
                else:
                    discriminator_id = self._forwarded_discriminator_id
                self._info_dicts[f"discriminator_{discriminator_id}"] = info_dict

                for indice, discriminator in enumerate(self.our_adapter_discriminators[self.adapter_name]):
                    if indice != discriminator_id:
                        info_dict = {
                            "running_mean": discriminator.running_mean,
                            "running_std": discriminator.running_std,
                            "num_batches_tracked": discriminator.num_batches_tracked,
                        }
                        self._info_dicts[f"discriminator_{indice}"] = info_dict

            # Use specified adapter for training
            return self.base_layer(x, _forwarded_adapter_id=self._forwarded_adapter_id)
        else:
            # during evaluation
            global STACK_FORWARD

            if STACK_FORWARD:
                losses, info_dicts = self._forward_discriminators_parallelly(x)
            else:
                losses, info_dicts = self._forward_discriminators(x)

            # Store info from all discriminators
            for indice, info_dict in enumerate(info_dicts):
                self._info_dicts[f"discriminator_{indice}"] = info_dict

            # Get the best discriminator for each sample
            top_1_idx_list = torch.argmin(losses, dim=0).tolist()

            self._info_dicts["losses"] = losses.transpose(0, 1)  # (n_discriminators, n_envs) -> (n_envs, n_discriminators)
            self._info_dicts["top_1_idx_list"] = top_1_idx_list

            # For each sample, get the adapter ID from its best discriminator
            adapter_results = []
            for idx, top_1_idx in enumerate(top_1_idx_list):
                adapter_id = self.our_adapter_discriminators[self.adapter_name][top_1_idx].connected_adapter_indices.item()
                if not self.module_config.batch_first:
                    # Handle time-first format
                    current_x = x[:, idx:idx+1]  # Select single sample while preserving dims
                else:
                    current_x = x[idx:idx+1]  # Select single sample while preserving dims
                
                # Forward through base layer with appropriate adapter
                current_result = self.base_layer(current_x, _forwarded_adapter_id=adapter_id)
                adapter_results.append(current_result)

            # Stack results back together
            if not self.module_config.batch_first:
                return torch.cat(adapter_results, dim=1)  # Concat along batch dimension (dim=1 for time-first)
            else:
                return torch.cat(adapter_results, dim=0)  # Concat along batch dimension (dim=0)

    def _forward_adapter(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # during training

            # forward specific discriminator
            if self._train_discriminator:
                _, info_dict = self.our_adapter_discriminators[self.adapter_name][self._forwarded_discriminator_id](x)

                if self._forwarded_discriminator_id == -1:
                    discriminator_id = len(self.our_adapter_discriminators[self.adapter_name]) - 1  
                else:
                    discriminator_id = self._forwarded_discriminator_id
                self._info_dicts[f"discriminator_{discriminator_id}"] = info_dict

                for indice, discriminator in enumerate(self.our_adapter_discriminators[self.adapter_name]):
                    if indice != discriminator_id:
                        info_dict = {
                            "running_mean" : discriminator.running_mean,
                            "running_std" : discriminator.running_std,
                            "num_batches_tracked" : discriminator.num_batches_tracked,
                        }
                        self._info_dicts[f"discriminator_{indice}"] = info_dict

            # forward specific adapter
            adapter_result = self.our_adapter_func_adapters[self.adapter_name][self._forwarded_adapter_id](x)
        else:
            # during evaluation

            global STACK_FORWARD

            if STACK_FORWARD:

                losses, info_dicts = self._forward_discriminators_parallelly(x)
                

                for indice, info_dict in enumerate(info_dicts):
                    self._info_dicts[f"discriminator_{indice}"] = info_dict

                batched_func_adapters = []

                top_1_idx_list = torch.argmin(losses, dim=0).tolist()

                self._info_dicts["losses"] = losses.transpose(0, 1) # (n_discriminators, n_envs) -> (n_envs, n_discriminators)
                self._info_dicts["top_1_idx_list"] = top_1_idx_list

                for top_1_idx in top_1_idx_list:
                    func_idx = self.our_adapter_discriminators[self.adapter_name][top_1_idx].connected_adapter_indices
                    batched_func_adapters.append(self.our_adapter_func_adapters[self.adapter_name][func_idx])

                params, buffers = stack_module_state(batched_func_adapters)
                prototype = batched_func_adapters[0]

                def forward_one_func_adapter(params_i, buffers_i, input_i):
                    y = functional_call(prototype, (params_i, buffers_i), (input_i,))
                    return y
                
                if not self.module_config.batch_first:
                    adapter_input = einops.rearrange(x, "t b d ... -> b t d ... ")
                else:
                    adapter_input = x
                
                adapter_result = vmap(forward_one_func_adapter)(params, buffers, adapter_input)

                if not self.module_config.batch_first:
                    adapter_result = einops.rearrange(adapter_result, "b t d ... -> t b d ... ")
            
            else:
                losses, info_dicts = self._forward_discriminators(x)

                for indice, info_dict in enumerate(info_dicts):
                    self._info_dicts[f"discriminator_{indice}"] = info_dict

                top_1_idx_list = torch.argmin(losses, dim=0).tolist()

                self._info_dicts["losses"] = losses.transpose(0, 1) # (n_discriminators, n_envs) -> (n_envs, n_discriminators)
                self._info_dicts["top_1_idx_list"] = top_1_idx_list

                # Process each sample individually
                adapter_results = []
                for idx, top_1_idx in enumerate(top_1_idx_list):
                    func_idx = self.our_adapter_discriminators[self.adapter_name][top_1_idx].connected_adapter_indices
                    
                    # Select single sample while preserving dims
                    if not self.module_config.batch_first:
                        current_x = x[:, idx:idx+1]  # time-first format
                    else:
                        current_x = x[idx:idx+1]  # batch-first format
                    
                    # Process this sample with its best adapter
                    current_result = self.our_adapter_func_adapters[self.adapter_name][func_idx](current_x)
                    adapter_results.append(current_result)

                # Stack results back together
                if not self.module_config.batch_first:
                    adapter_result = torch.cat(adapter_results, dim=1)  # Concat along batch dimension (dim=1 for time-first)
                else:
                    adapter_result = torch.cat(adapter_results, dim=0)  # Concat along batch dimension (dim=0)

        return adapter_result

    def _forward_discriminators(self, x: torch.Tensor):

        losses = []
        info_dicts = []

        for discriminator in self.our_adapter_discriminators[self.adapter_name]:
            loss, info_dict = discriminator(x)
            losses.append(loss)
            info_dicts.append(info_dict)

        losses = torch.stack(losses, dim=0)

        return losses, info_dicts


    def _forward_discriminators_parallelly(self, x: torch.Tensor):
        params, buffers = stack_module_state(self.our_adapter_discriminators[self.adapter_name])
        prototype = self.our_adapter_discriminators[self.adapter_name][0]

        def forward_one_discriminator(params_i, buffers_i):
            loss, info_dict = functional_call(prototype, (params_i, buffers_i), (x,))
            vals = tuple(info_dict[k] for k in info_dict.keys())
            return loss, vals

        losses, vals = vmap(forward_one_discriminator)(params, buffers)
        
        keys = self.our_adapter_discriminators[self.adapter_name][0].info_dict_keys

        info_dicts = []

        for batch_indice in range(self.num_discriminators):
            info_dict = {}
            for indice, key in enumerate(keys):
                info_dict[key] = vals[indice][batch_indice]
            info_dicts.append(info_dict)

        return losses, info_dicts

    def _create_adapter(self) -> FuncAdapterWrapper:
        if self.use_lora:
            # For LoRA, create a dummy adapter that will hold references to LoRA params
            adapter = nn.Module()
            # Will be populated with lora_adapter attribute later
        elif self.module_config.use_trainable_copy:
            adapter = copy.deepcopy(self.base_layer)
        else:
            adapter = FuncAdapter(
                self.module_config.func_adapter_cfg, 
                self.module_config.feature_dim, 
                self.module_config.out_feature_dim
            )
        for p in adapter.parameters():
            p.requires_grad = True
        return FuncAdapterWrapper(self.module_config, adapter)
    
    def _create_lora_adapter(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                pass

    def _create_discriminator(self):
        disc_cls = get_discriminaor_class(self.module_config.discriminator_cfg.type)
        return disc_cls(self.module_config.discriminator_cfg, self.module_config.feature_dim)

    def add_adapter_and_discriminator(self, new_task_id:int):
        # Create new adapter wrapper in both cases
        new_adapter = self._create_adapter()
        new_adapter.func_adapter.task_id = torch.tensor(new_task_id, dtype=torch.int64)
        new_adapter.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        self.our_adapter_func_adapters[self.adapter_name].append(new_adapter)

        if self.use_lora:
            # For LoRA, add new adapter to all wrapped linear layers
            self.base_layer.add_adapter(new_task_id)
            # Store reference to all LoRA modules for this adapter
            lora_adapter = {}
            for name, lora_module in self.base_layer.lora_modules.items():
                lora_adapter[name] = {
                    'lora_A': lora_module.lora_a[str(new_task_id)],
                    'lora_B': lora_module.lora_b[str(new_task_id)]
                }
            # Store in our standard adapter structure
            new_adapter.lora_adapter = lora_adapter
            # Get parameters from all LoRA modules
            adapter_parameter = []
            for lora_module in self.base_layer.lora_modules.values():
                adapter_parameter.append(lora_module.lora_a[str(new_task_id)])
                adapter_parameter.append(lora_module.lora_b[str(new_task_id)])
        else:
            # Original case
            adapter_parameter = list(new_adapter.parameters())

        self.num_adapters += 1
        discriminator_parameter = self.add_discriminator(self.num_adapters - 1, new_task_id)

        return adapter_parameter, discriminator_parameter

    def add_discriminator(self, connected_adapter_indices:int, new_task_id:int):
        new_discriminator = self._create_discriminator()
        new_discriminator.task_id = torch.tensor(new_task_id, dtype=torch.int64)
        new_discriminator.connected_adapter_indices = torch.tensor(connected_adapter_indices, dtype=torch.int64)
        new_discriminator.connected_adapter_task_id = self.our_adapter_func_adapters[self.adapter_name][connected_adapter_indices].func_adapter.task_id
        new_discriminator.to(device=self._base_layer_device, dtype=self._base_layer_dtype)
        self.our_adapter_discriminators[self.adapter_name].append(new_discriminator)
        self.num_discriminators += 1

        discriminator_parameter = list(new_discriminator.parameters())

        return discriminator_parameter
    
    def train_discriminator(self, train_discriminator:bool):
        self._train_discriminator = train_discriminator

    def track_z_score(self, require_z_score:bool):
        for discriminator in self.our_adapter_discriminators[self.adapter_name]:
            discriminator.require_z_score = require_z_score

    def update_stats(self, require_update_stats:bool):
        self.our_adapter_discriminators[self.adapter_name][self._forwarded_discriminator_id].require_update_stats = require_update_stats
        # for discriminator in self.our_adapter_discriminators[self.adapter_name]:
        #     discriminator.require_update_stats = require_update_stats

    def get_adapter_id_by_discriminator_id(self, discriminator_id):
        return self.our_adapter_discriminators[self.adapter_name][discriminator_id].connected_adapter_indices.item()
    
    @property
    def info_dicts(self):
        return self._info_dicts