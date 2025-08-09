from __future__ import annotations
from dataclasses import dataclass, field
import draccus
from typing import Any, List, Optional, Union
import re

from peft.config import PeftConfig
from peft.utils.peft_types import PeftType

ModuleSelector = Union[str, "re.Pattern[str]"]


@dataclass
class DiscriminatorConfig(draccus.ChoiceRegistry):
    """
    This is the sub-configuration class to store the configuration of a [`OurAdapterModel`].
    
    Args:
        feature_dim (`int`):
            The dimension of the input feature. Given an input of shape (B, T, D), D is feature_dim
        max_batches_tracked (`int`):
            How many batches will be tracked to calculate the statistic.
    """
    feature_dim: int = None
    max_batches_tracked: int = 2000

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "autoencoder"


@dataclass
class FuncAdapterConfig(draccus.ChoiceRegistry):
    """
    This is the sub-configuration class to store the configuration of a [`OurAdapterModel`].
    
    Args:
        use_trainable_copy (`bool`):
            whether to copy the module from base model as adapter or not
        use_lora (`bool`):
            whether to use lora on functional adapter or not
        lora_rank (`int`):
            Lora attention dimension (the "rank").
    """

@dataclass
class OurAdapterConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`OurAdapterModel`].

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen (if
            the model is a PreTrainedModel, the output layer excluded). If this is not specified, modules will be
            chosen according to the model architecture. If the architecture is not known, an error will be raised -- in
            this case, you should specify the target modules manually. To avoid targeting any modules (because you want
            to apply `target_parameters`), set `target_modules=[]`.
        discriminator_cfg (`DiscriminatorConfig`):
            The configuration of Discriminator
        func_adapter_cfg (`FuncAdapterConfig`):
             The configuration of FuncAdapter
    """
    target_modules: Optional[List[ModuleSelector]] = None
    inference_mode: bool = False
    task_type: Optional[str] = None
    modules_to_save: Optional[List[str]] = None

    discriminator_cfg: DiscriminatorConfig = None
    func_adapter_cfg: FuncAdapterConfig = None

    _default_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
        "fc1", "fc2", "wo", "wi", "w1", "w2",
    ])

    def __post_init__(self) -> None:
        super().__post_init__()
        # Assign to a valid PEFT type so load/save works. It does not alter the tuner name.
        self.peft_type = PeftType.OUR_ADAPTER
        # Provide sensible defaults if user doesn’t supply target_modules
        if self.target_modules is None:
            self.target_modules = list(self._default_targets)
        # Allow `re:` prefix to specify regex
        compiled: List[ModuleSelector] = []
        for sel in self.target_modules:
            if isinstance(sel, str) and sel.startswith("re:"):
                compiled.append(re.compile(sel[3:]))
            else:
                compiled.append(sel)
        self.target_modules = compiled
