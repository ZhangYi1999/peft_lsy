import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from .config import FuncAdapterConfig

class LoRALayer(nn.Module):
    """LoRA wrapper for Linear and Conv modules"""
    def __init__(self, base_layer: Union[nn.Linear, nn.Conv1d, nn.Conv2d], config: FuncAdapterConfig):
        super().__init__()
        self.base_layer = base_layer
        self.r = config.lora_rank
        self.alpha = float(config.lora_alpha)  # Convert to float in case it's an int
        
        # LoRA matrices for different tasks
        self.lora_a = nn.ParameterDict({})  # {task_id: down_proj}
        self.lora_b = nn.ParameterDict({})  # {task_id: up_proj}
        
        # Optional dropout
        self.lora_dropout = nn.Identity() # TODO: nn.Dropout(p=config.dropout_p) if config.dropout_p > 0 else 
        
        # For scaling the LoRA output
        self.scaling = self.alpha / self.r
        
        # Get input and output features based on layer type
        if isinstance(base_layer, nn.Linear):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
            self.is_conv = False
        else:  # Conv layers
            self.in_features = base_layer.in_channels
            self.out_features = base_layer.out_channels
            self.is_conv = True
            self.kernel_size = base_layer.kernel_size
            self.stride = base_layer.stride
            self.padding = base_layer.padding
            self.groups = base_layer.groups
            self.conv_op = base_layer.__class__  # Store Conv1d or Conv2d class
        
        # Keep track of number of tasks/adapters
        self.num_adapters = 0
        
    def add_adapter(self, task_id: int):
        """Add a new LoRA adapter for a task"""
        # Convert task_id to string since ParameterDict requires string keys
        task_id_str = str(task_id)
        
        if task_id_str not in self.lora_a:
            if self.is_conv:
                # For conv layers, we need to consider the kernel size
                # lora_a is the down projection: (r, in_channels, *kernel_size)
                # lora_b is the up projection: (out_channels, r, 1, ..., 1)
                self.lora_a[task_id_str] = nn.Parameter(
                    torch.zeros((self.r, self.in_features, *self.kernel_size), 
                              device=self.base_layer.weight.device, requires_grad=True)
                )
                self.lora_b[task_id_str] = nn.Parameter(
                    torch.zeros((self.out_features, self.r, *(1,) * len(self.kernel_size)), 
                              device=self.base_layer.weight.device, requires_grad=True)
                )
            else:
                # For linear layers
                self.lora_a[task_id_str] = nn.Parameter(
                    torch.zeros((self.r, self.in_features), device=self.base_layer.weight.device, requires_grad=True)
                )
                self.lora_b[task_id_str] = nn.Parameter(
                    torch.zeros((self.out_features, self.r), device=self.base_layer.weight.device, requires_grad=True)
                )
            
            # Initialize with small random values
            nn.init.kaiming_uniform_(self.lora_a[task_id_str], a=math.sqrt(5))
            nn.init.zeros_(self.lora_b[task_id_str])
            
            self.num_adapters += 1
            
    def forward(self, x: torch.Tensor, adapter_id: int = -1) -> torch.Tensor:
        # Regular transformation
        base_output = self.base_layer(x)
        
        # If no adapter_id specified or invalid, return base output
        if adapter_id < -1:
            return base_output

        if adapter_id == -1:
            adapter_id = len(self.lora_a) - 1
            
        # LoRA transformation
        adapter_id_str = str(adapter_id)
        if self.is_conv:
            # For conv layers, we need to use conv operations
            dropped_x = self.lora_dropout(x)
            
            # Down projection
            down = F.conv1d(dropped_x, self.lora_a[adapter_id_str], 
                          stride=self.stride, padding=self.padding, 
                          groups=self.groups) if isinstance(self.base_layer, nn.Conv1d) else \
                   F.conv2d(dropped_x, self.lora_a[adapter_id_str], 
                          stride=self.stride, padding=self.padding, 
                          groups=self.groups)
            
            # Up projection
            lora_output = F.conv1d(down, self.lora_b[adapter_id_str]) if isinstance(self.base_layer, nn.Conv1d) else \
                         F.conv2d(down, self.lora_b[adapter_id_str])
        else:
            # For linear layers
            lora_output = self.lora_dropout(x) @ self.lora_a[adapter_id_str].t() @ self.lora_b[adapter_id_str].t()
        
        # Combine with scaling
        return base_output + self.scaling * lora_output