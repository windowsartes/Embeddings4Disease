import torch
import typing as tp
from dataclasses import dataclass


@dataclass
class TrainingArgs:
    mode: str
    learning_rate: float
    backbone_learning_rate: tp.Optional[float]
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    n_epochs: int
    n_warmup_epochs: int
    criterion: tp.Type[torch.nn.modules.loss._Loss]
    batch_size: int
    device: torch.device
