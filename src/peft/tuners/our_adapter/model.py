# peft/tuners/our_adapter/model.py

from __future__ import annotations

import re
from typing import Any, List, Union

import torch.nn as nn
from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists

from .config import OurAdapterConfig
from .layer import OurAdapterLayer

def _matches_target(name: str, selectors: List[Union[str, "re.Pattern[str]"]]) -> bool:
    for sel in selectors:
        if isinstance(sel, str):
            if sel == name or name.endswith(f".{sel}") or sel in name.split("."):
                return True
        else:
            if sel.search(name):
                return True
    return False

class OurAdapterModel(BaseTuner):
    """
    PEFT-compatible tuner that injects OurAdapterLayer into target modules.
    """
    # ------- BaseTuner callbacks --------
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
    ):
        if not _matches_target(current_key, peft_config.target_modules or []):
            return
        wrapper = OurAdapterLayer(
            model=self,
            target=target,
            target_forward=target.forward,
            layer_number=len(self._our_adapter_layers),
            config=peft_config,
        )
        target.forward = wrapper.forward  # type: ignore[method-assign]
        self._our_adapter_layers.append(wrapper)

    # Add this static method: return the config unchanged
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """
        Prepare and return the adapter config.
        For OurAdapter, we simply return the config unchanged.
        """
        return peft_config

    def _mark_only_adapters_as_trainable(self):
        """
        Mark only the adapter parameters as trainable.
        Freeze everything else (already done in __init__).
        This satisfies the BaseTuner abstract method contract.
        """
        # Ensure base model parameters remain frozen
        for param in self.model.parameters():
            param.requires_grad = False

        # Explicitly enable gradients on adapter params
        for layer in getattr(self, "_our_adapter_layers", []):
            for p in layer.adapter.trainable_parameters():
                p.requires_grad = True

    # ------- Convenience controls --------
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def enable_adapter_layers(self) -> None:
        self.disabled = False

    def disable_adapter_layers(self) -> None:
        self.disabled = True

    def set_active_task(self, task_idx: int) -> None:
        for layer in self._our_adapter_layers:
            layer.adapter.set_active_task(task_idx)

    def set_forward_all(self) -> None:
        for layer in self._our_adapter_layers:
            layer.adapter.set_forward_all()

    def set_forward(self, block_id: int, disc_id: int) -> None:
        for layer in self._our_adapter_layers:
            layer.adapter.set_forward(block_id, disc_id)

    def add_discriminators(self, block_id: int, num_discriminators: int) -> None:
        for layer in self._our_adapter_layers:
            layer.adapter.add_discriminators(block_id, num_discriminators)

    def add_adapters(self, num_blocks: int) -> None:
        for layer in self._our_adapter_layers:
            layer.adapter.add_adapters(num_blocks)

    def trainable_parameters(self):
        params = []
        for layer in self._our_adapter_layers:
            params += layer.adapter.trainable_parameters()
        return params

    