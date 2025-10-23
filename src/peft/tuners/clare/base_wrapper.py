import torch
import torch.nn as nn
from typing import Dict, Union, Any
from .lora_adapter import LoRALayer

class LoRABaseWrapper(nn.Module):
    """Wrapper for base model to handle LoRA adapters"""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.lora_modules: Dict[str, LoRALayer] = {}
        
    def wrap_linear_layers(self, config):
        """Recursively wrap all Linear and Conv layers with LoRA adapters"""
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Replace the module with LoRA wrapped version
                lora_layer = LoRALayer(module, config)
                # Store reference to LoRA layer
                self.lora_modules[name] = lora_layer
                # Replace the original layer in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = self.base_model.get_submodule(parent_name)
                    setattr(parent, child_name, lora_layer)
                else:
                    setattr(self.base_model, child_name, lora_layer)
                    
    def add_adapter(self, task_id: int):
        """Add a new adapter to all LoRA layers"""
        for lora_layer in self.lora_modules.values():
            lora_layer.add_adapter(task_id)
            
    def forward(self, *args, _forwarded_adapter_id: int = -1, **kwargs):
        """Forward pass with specified adapter ID"""
        # Store original adapter ID
        self._cur_adapter_id = _forwarded_adapter_id
        
        # If using LoRA (_forwarded_adapter_id >= 0), we need to ensure the adapter ID
        # is passed to all LoRA layers during the forward pass
        if _forwarded_adapter_id >= 0:
            # The LoRA layers will automatically handle the adapter_id in their forward
            output = self.base_model(*args, **kwargs)
        else:
            # Regular forward pass without LoRA
            output = self.base_model(*args, **kwargs)
            
        return output
        
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base_model if not found in wrapper"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)