import os
import platform
import typing as tp

import numpy as np
import pytorch_warmup as warmup
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from embeddings4disease.callbacks.custom_callbacks import CustomCallback
from embeddings4disease.trainer.training_args import TrainingArgs
from embeddings4disease.trainer.training_state import TrainingState


class Trainer:
    """
    Base class for the custom head training. Use it in the case there is no useful trainer in hugging face library.

    Args:
        model (nn.Module): model you want to train.
        train_dataloader (DataLoader): dataloader with training data.
        eval_dataloader (DataLoader): dataloader with validation data.
        callbacks (list[custom_callbacks.CustomCallback]): list of callback you want to use during the trainig.
        args (TrainingArgs): trainig args: criterion, hyperparameters and etc.
    """
    def __init__(
        self,
        model: nn.Module,
        data_collator: tp.Callable,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        callbacks: list[CustomCallback],
        args: TrainingArgs,
    ):
        self.__model: nn.Module = model

        self.__train_dataloader: DataLoader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            drop_last=True,
        )

        self.__eval_dataloader: DataLoader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            drop_last=False,
        )

        self.__callbacks: list[CustomCallback] = callbacks

        self.__training_args: TrainingArgs = args

        self.__training_state: TrainingState = TrainingState()

    def _create_optimizer(self) -> optim.Optimizer:
        """
        Creates an AdamW optimizer with the parameters, specified in trainig_args.

        Returns:
            optim.Optimizer: your model's AdamW optimizer.
        """
        if self.__training_args.mode == "transfer learning":
            optimizer: optim.AdamW = optim.AdamW(
                self.__model.head.parameters(),
                lr=self.__training_args.learning_rate,
                betas=(self.__training_args.adam_beta1, self.__training_args.adam_beta2),
                weight_decay=self.__training_args.weight_decay,
            )
        elif self.__training_args.mode == "fine tuning":
            optimizer = optim.AdamW(
                [
                    {
                        "params": self.__model.backbone.parameters(),
                        "lr": self.__training_args.backbone_learning_rate,
                    },
                    {"params": self.__model.head.parameters()},
                ],
                lr=self.__training_args.learning_rate,
                betas=(self.__training_args.adam_beta1, self.__training_args.adam_beta2),
                weight_decay=self.__training_args.weight_decay,
            )
        return optimizer

    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
        """
        Creates a cosine scheduler for the given optimizer. Its parameters are specified in the training args.

        Args:
            optimizer (optim.Optimizer): optimizer you want to schedule.

        Returns:
            optim.lr_scheduler.LRScheduler: linear scheduler for your optimizer.
        """
        T_max: int = self.__training_args.n_epochs - self.__training_args.n_warmup_epochs

        scheduler: optim.lr_scheduler.CosineAnnealingLR = (
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        )

        return scheduler

    def _create_warmup_scheduler(self, optimizer: optim.Optimizer) -> warmup.BaseWarmup:
        """
        Creates a linear warmup scheduler for the given optimizer. Its parameters are specified in the training_args.

        Args:
            optimizer (optim.Optimizer): optimizer you want to warmup.

        Returns:
            warmup.BaseWarmup: warmup scheduler for your optimizer.
        """
        warmup_period: int = self.__training_args.n_warmup_epochs

        warmup_scheduler: warmup.LinearWarmup = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)

        return warmup_scheduler

    def train(self) -> None:
        """
        Main method for the model training. For now there is 4 stages: train step, eval_step, 'on_evaluation' on which
        callback's 'on_evaluation' method is called, 'on_save' on which callbacks's 'on_save' method is called.
        """
        self.__model = self.__model.to(self.__training_args.device)

        optimizer: optim.Optimizer = self._create_optimizer()
        scheduler: optim.lr_scheduler.LRScheduler = self._create_scheduler(optimizer)
        warmup_scheduler: warmup.BaseWarmup = self._create_warmup_scheduler(optimizer)

        criterion: torch.nn.modules.loss._Loss = self.__training_args.criterion()

        progress_bar = tqdm(range(self.__training_args.n_epochs))
        for epoch in progress_bar:
            progress_bar.set_description(f"epoch #{epoch}")

            average_train_loss: float = self._train_step(
                optimizer=optimizer,
                scheduler=scheduler,
                warmup_scheduler=warmup_scheduler,
                dataloader=self.__train_dataloader,
                criterion=criterion,
            )

            self.__training_state.train_loss_history[epoch] = average_train_loss

            average_eval_loss: float = self._eval_step(
                dataloader=self.__eval_dataloader,
                criterion=criterion,
            )

            self.__training_state.eval_loss_history[epoch] = average_eval_loss

            for callback in self.__callbacks:
                callback.on_evaluate(
                    self.__training_state,
                    self.__training_args,
                    self.__model,
                    **{"dataloader": self.__eval_dataloader},
                )

            for callback in self.__callbacks:
                callback.on_save(
                    self.__training_state,
                    self.__model,
                    optimizer,
                )

            self.__training_state.epoch += 1

            os.system("cls" if platform.system() == "Windows" else "clear")

    def _train_step(
        self,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        warmup_scheduler: warmup.BaseWarmup,
        dataloader: DataLoader,
        criterion: torch.nn.modules.loss._Loss,
    ) -> float:
        """
        One trainig step.

        Args:
            optimizer (optim.Optimizer): your model's optimizer.
            scheduler (optim.lr_scheduler.LRScheduler): optimizer's scheduler.
            warmup_scheduler (warmup.BaseWarmup): optimizer's warmup scheduler.
            dataloader (DataLoader): dataloader you want to use on training.
            criterion (torch.nn.modules.loss._Loss): criterion function you want to use.

        Returns:
            float: average loss during the trainig step.
        """
        self.__model.train()

        losses: list[float] = []

        device: torch.device = self.__training_args.device
        n_warmup_epochs: int = self.__training_args.n_warmup_epochs

        progress_bar = tqdm(dataloader)
        for input_tensor, target_tensor in progress_bar:
            progress_bar.set_description("Training step")

            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            predicted_tensor: torch.Tensor = self.__model(input_tensor)

            optimizer.zero_grad()

            loss: torch.Tensor = criterion(predicted_tensor, target_tensor)
            loss.backward()  # type: ignore

            optimizer.step()

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= n_warmup_epochs:
                    scheduler.step()

            losses.append(loss.item())

        progress_bar.close()

        return float(np.mean(losses))

    @torch.no_grad()
    def _eval_step(
        self,
        dataloader: DataLoader,
        criterion: torch.nn.modules.loss._Loss,
    ) -> float:
        """
        One training step.

        Args:
            dataloader (DataLoader): dataloader you want to use on validation.
            criterion (torch.nn.modules.loss._Loss): criterion you want to use. Note that backward() won't be called.

        Returns:
            float: average loss during the validation step.
        """
        self.__model.eval()

        losses: list[float] = []

        device: torch.device = self.__training_args.device

        progress_bar = tqdm(dataloader)
        for input_tensor, target_tensor in progress_bar:
            progress_bar.set_description("Evaluation step")

            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            predicted_tensor: torch.Tensor = self.__model(input_tensor)

            loss: torch.Tensor = criterion(predicted_tensor, target_tensor)

            losses.append(loss.item())
        progress_bar.close()

        return float(np.mean(losses))
