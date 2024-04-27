from dataclasses import dataclass, field

@dataclass
class TrainingState:
    train_loss_history: dict[int, float] = field(default_factory=dict)
    eval_loss_history: dict[int, float] = field(default_factory=dict)
