# peft/tuners/our_adapter/model.py

from __future__ import annotations

import re
from typing import Any, List, Union, Optional, Tuple

import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists

from .config import OurAdapterConfig
from .layer import OurAdapterLayer

def extract_layer(current_key: str, key_pattern: str) -> Optional[Tuple[str, int]]:
    """
    Returns (layer_name, layer_id) if `key_pattern` matches `current_key`,
    else returns None.

    key_pattern should contain:
      - (?P<layer_name>...)   -> e.g. (layers) or (encoders|decoders)
      - (?P<layer_id>\d+)     -> the numeric id
    """
    m = re.search(key_pattern, current_key)
    if not m:
        return None
    layer_name = m.group("layer_name")
    layer_id = int(m.group("layer_id"))
    return layer_name, layer_id


class OurAdapterModel(BaseTuner):
    """
    PEFT-compatible tuner that injects OurAdapterLayer into target modules.
    """

    _our_adapter_layers: List[OurAdapterLayer] = []

    @staticmethod
    def _check_target_module_exists(peft_config: OurAdapterConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)
    
    def _create_and_replace(
        self,
        peft_config: OurAdapterConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        *,
        parameter_name: Optional[str] = None,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        else:
            target_modules = peft_config.target_modules
            if isinstance(target_modules, str):
                layer_name, layer_id = extract_layer(current_key, peft_config.target_modules)
            else:
                layer_name, layer_id = extract_layer(current_key, rf"(?P<layer_name>.+)\.(?P<layer_id>\d+)(?:\.[^.]+)*\.{re.escape(target_name)}$")

        # normal situation
        device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
        new_module = self._create_new_module(peft_config, adapter_name, target, layer_name, layer_id, device_map=device_map)
        self._replace_module(parent, target_name, new_module, target)


    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # meta = torch.device("meta")
        # # dispatch to correct device
        # for name, module in new_module.named_modules():
        #     if self.prefix in name:
        #         weight = next(child.parameters())
        #     if not any(p.device == meta for p in module.parameters()):
        #         module.to(weight.device)

    @staticmethod
    def _create_new_module(peft_config, adapter_name, target, layer_name, layer_id, **kwargs):
        
        if layer_name in peft_config.structure:
            layer_structure = peft_config.structure[layer_name]
        else:
            layer_structure = []
        
        new_module = OurAdapterLayer(
            base_layer=target,
            peft_config=peft_config,
            adapter_name=adapter_name,
            layer_name=layer_name, 
            layer_id=layer_id,
            layer_structure=layer_structure
        )

        return new_module

    def disable_adapter_layers(self):
        pass

    def enable_adapter_layers(self):
        pass

    def set_adapter(self, adapter_name: str | list[str]):
        self.active_adapter = adapter_name

    # Add this static method: return the config unchanged
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """
        Prepare and return the adapter config.
        For OurAdapter, we simply return the config unchanged.
        """
        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Mark only the adapter parameters as trainable.
        Freeze everything else (already done in __init__).
        This satisfies the BaseTuner abstract method contract.
        """
        for n, p in model.named_parameters():
            p.requires_grad = False

        # Explicitly enable gradients on adapter params
        for layer in self._our_adapter_layers:
            for p in layer.parameters():
                p.requires_grad = True

    # ------- Convenience controls --------
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    