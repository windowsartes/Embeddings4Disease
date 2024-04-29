from dataclasses import dataclass, field

import numpy as np

@dataclass
class TrainingState:
    train_loss_history: dict[int, float] = field(default_factory=dict)
    eval_loss_history: dict[int, float] = field(default_factory=dict)
    eval_loss_best_value: float = np.inf
    epoch: int = 0
