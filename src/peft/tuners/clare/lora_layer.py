import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from .config import FuncAdapterConfig

class LoRALayer(nn.Module):
    """LoRA wrapper for Linear and Conv modules"""
    def __init__(self, original_layer: nn.Module, config: FuncAdapterConfig):
        super().__init__()
        self.original_layer = original_layer
        self.r = config.lora_rank
        self.alpha = config.lora_alpha
        self.lora_dropout = nn.Dropout(p=config.dropout_p) if hasattr(config, "dropout_p") else nn.Identity()
        self.scaling = self.alpha / self.r
        
        if isinstance(original_layer, nn.Linear):
            in_features, out_features = original_layer.in_features, original_layer.out_features
        else:
            raise NotImplementedError("The LoRA adapter only supports nn.Linear for now.")

        self.lora = {
            "A": None,
            "B": None
        }

        self.in_features = in_features
        self.out_features = out_features
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original_layer(x)

        if self.lora["A"] is None:
            raise ValueError("Uninitialized lora_a.")
        else:
            lora_a = self.lora["A"]

        if self.lora["B"] is None:
            raise ValueError("Uninitialized lora_b.")
        else:
            lora_b = self.lora["B"]
    
        lora_output = lora_b(lora_a(self.lora_dropout(x))) * self.scaling

        return base_output + lora_output