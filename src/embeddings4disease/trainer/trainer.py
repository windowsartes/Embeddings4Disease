import numpy as np
import pytorch_warmup as warmup
import torch
import torch.nn as nn
import typing as tp
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from embeddings4disease.trainer.training_args import TrainingArgs
from embeddings4disease.trainer.training_state import TrainingState


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        callbacks: list[tp.Any],  # just for now
        args: TrainingArgs,
    ):
        self.model: nn.Module = model

        self.train_dataloader: DataLoader = train_dataloader
        self.eval_dataloader: DataLoader = eval_dataloader

        self.callbacks = callbacks

        self.training_args: TrainingArgs = args

        self.training_state: TrainingState = TrainingState()

    def _create_optimizer(self) -> optim.Optimizer:
        if self.training_args.mode == "transfer learning":
            optimizer: optim.AdamW = optim.AdamW(
                self.model.head.parameters(),
                lr=self.training_args.learning_rate,
                betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
                weight_decay=self.training_args.weight_decay,
            )
        elif self.training_args.mode == "fine tuning":
            optimizer = optim.AdamW(
                [
                    {
                        "params": self.model.backbone.parameters(),
                        "lr": self.training_args.backbone_learning_rate,
                    },
                    {"params": self.model.head.parameters()},
                ],
                lr=self.training_args.learning_rate,
                betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
                weight_decay=self.training_args.weight_decay,
            )
        return optimizer

    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
        T_max: int = self.training_args.n_epochs - self.training_args.n_warmup_epochs

        scheduler: optim.lr_scheduler.CosineAnnealingLR = (
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        )

        return scheduler

    def _create_warmup_scheduler(self, optimizer: optim.Optimizer) -> warmup.BaseWarmup:
        warmup_period: int = self.training_args.n_warmup_epochs

        warmup_scheduler: warmup.LinearWarmup = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)

        return warmup_scheduler

    def train(self) -> None:
        self.model = self.model.to(self.training_args.device)

        optimizer: optim.Optimizer = self._create_optimizer()
        scheduler: optim.lr_scheduler.LRScheduler = self._create_scheduler(optimizer)
        warmup_scheduler: warmup.BaseWarmup = self._create_warmup_scheduler(optimizer)

        criterion: torch.nn.modules.loss._Loss = self.training_args.criterion()

        for epoch in range(self.training_args.n_epochs):
            average_train_loss = self._train_step(
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 warmup_scheduler=warmup_scheduler,
                                                 dataloader=self.train_dataloader,
                                                 criterion=criterion,
                                                )
            self.training_state.train_loss_history[epoch] = average_train_loss

            average_eval_loss = self._eval_step(
                                               dataloader=self.eval_dataloader,
                                               criterion=criterion,
                                              )

            self.training_state.eval_loss_history[epoch] = average_eval_loss

    def _train_step(self,
                    optimizer: optim.Optimizer,
                    scheduler: optim.lr_scheduler.LRScheduler,
                    warmup_scheduler: warmup.BaseWarmup,
                    dataloader: DataLoader,
                    criterion: torch.nn.modules.loss._Loss,
                   ) -> float:
        self.model.train()

        losses: list[float] = []

        device: torch.device = self.training_args.device
        n_warmup_epochs: int = self.training_args.n_warmup_epochs

        for input_tensor, target_tensor in tqdm(dataloader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            predicted_tensor: torch.Tensor = self.model(input_tensor)

            optimizer.zero_grad()

            loss: torch.Tensor = criterion(predicted_tensor, target_tensor)
            loss.backward()  # type: ignore

            optimizer.step()

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= n_warmup_epochs:
                    scheduler.step()

            losses.append(loss.item())

        return float(np.mean(losses))

    @torch.no_grad()
    def _eval_step(self,
                   dataloader: DataLoader,
                   criterion: torch.nn.modules.loss._Loss,
                  ) -> float:
        self.model.eval()

        losses: list[float] = []

        device: torch.device = self.training_args.device

        for input_tensor, target_tensor in tqdm(dataloader):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            predicted_tensor: torch.Tensor = self.model(input_tensor)

            loss: torch.Tensor = criterion(predicted_tensor, target_tensor)

            losses.append(loss.item())

        return float(np.mean(losses))
