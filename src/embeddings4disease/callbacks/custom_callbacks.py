import json
import pathlib

import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from embeddings4disease.metrics.multilabel_head_metrics import MultiLabelHeadMetricComputer
from embeddings4disease.trainer.training_args import TrainingArgs
from embeddings4disease.trainer.training_state import TrainingState
from embeddings4disease.utils import utils


class CustomCallback:
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        pass

    def on_evaluate(self,  # type: ignore
                    training_state: TrainingState,
                    training_args: TrainingArgs,
                    model: nn.Module,
                    **kwargs,
                   ) -> None:
        pass

    def on_save(self,  # type: ignore
                training_state: TrainingState,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                **kwargs,
               ) -> None:
        pass


class MetricComputerCallback(CustomCallback):
    def __init__(  # type: ignore
            self,
            metrics_storage_dir: str | pathlib.Path,
            use_metrics: dict[str, bool],
            device: torch.device,
            period: int = 10,
            threshold: float = 0.5,
            save_plot: bool = True,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)

        self.metrics_storage_dir: pathlib.Path = pathlib.Path(metrics_storage_dir)
        utils.create_dir(self.metrics_storage_dir)

        self.period: int = period

        self.use_metrics: dict[str, bool] = use_metrics
        self.save_plot: bool = save_plot

        self.threshold: float = threshold
        self.device: torch.device = device

        if self.save_plot:
            sns.set_style("white")
            sns.color_palette("bright")

        for metric, usage in self.use_metrics.items():
            if usage:
                self.__initialize_metric_storage(metric)

    def __initialize_metric_storage(self, metric_name: str) -> None:
        """
        For every metric there is its own file; this method initialize it with empty object.

        Args:
            metric_name (str): metric's name; it will be used as a filename.
        """
        self.__dump_logs(metric_name, {})

    def __dump_logs(self, metric_name: str, logs: dict[str, float]) -> None:
        """
        Dumps metric's loggs to its log file.

        Args:
            metric_name (str): metric's name.
            logs (dict[int, float]): metric's logs.
        """
        utils.create_dir(pathlib.Path(self.metrics_storage_dir).joinpath(
            pathlib.Path(metric_name)))

        file_name: pathlib.Path = pathlib.Path(self.metrics_storage_dir).joinpath(
            pathlib.Path(metric_name).joinpath(f"{metric_name}.json")
        )

        with open(file_name, "w") as file:
            json.dump(logs, file)

    def __load_logs(self, metric_name: str) -> dict[str, float]:
        """
        Loads metric's logs from its log file.

        Args:
            metric_name (str):  metric's name.

        Returns:
            dict[str, float]: metric's logs.
        """
        file_name: pathlib.Path = pathlib.Path(self.metrics_storage_dir).joinpath(
            pathlib.Path(metric_name).joinpath(f"{metric_name}.json")
        )

        with open(file_name, "r") as file:
            logs: dict[str, float] = json.load(file)

        return logs

    def on_evaluate(self,  # type: ignore
                    training_state: TrainingState,
                    training_args: TrainingArgs,
                    model: nn.Module,
                    **kwargs,
                   ) -> None:
        """
        Computes metrics' values during the validation. Optionall this method will draw a plots with metrics values
        during all the evaluation steps.

        Args:
            training_state (TrainingState): training state. It stores loss history, current epoch number and etc.
            trainihg_args (TrainingArgs): training arg that was created by the factory.
            model (nn.Module): model you want to evaluate.
        """
        if training_state.epoch % self.period == 0:
            metric_computer: MultiLabelHeadMetricComputer = MultiLabelHeadMetricComputer(
                self.threshold,
                kwargs["dataloader"],
                self.device,
            )

            metrics: dict[str, float] = metric_computer.get_metrics_value(
                model, self.use_metrics
            )

            for metric_name, usage in self.use_metrics.items():
                if usage:
                    logs = self.__load_logs(metric_name)
                    logs[str(training_state.epoch)] = metrics[metric_name]

                    self.__dump_logs(metric_name, logs)

                    if self.save_plot:
                        sns.set()

                        x_ticks: list[int] = [
                            int(float(value)) for value in list(logs.keys())
                        ]
                        values: list[float] = list(logs.values())

                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)

                        ax.yaxis.set_minor_locator(AutoMinorLocator())

                        ax.tick_params(which="major", length=7)
                        ax.tick_params(which="minor", length=4, color="r")

                        ax.grid(which="minor", alpha=0.2)
                        ax.grid(which="major", alpha=0.5)

                        plt.xticks(rotation=45)

                        ax.plot(x_ticks, values)

                        plt.title(f"Value of {metric_name} on validation")
                        plt.xlabel("Epoch")

                        plt.savefig(
                            self.metrics_storage_dir.joinpath(
                                pathlib.Path(metric_name).joinpath(f"{metric_name}.png")
                            )
                        )


class SaveLossHistoryCallback(CustomCallback):
    def __init__(self, loss_storage_dir: str | pathlib.Path, save_plot: bool):
        super().__init__()

        self.loss_storage_dir: pathlib.Path = pathlib.Path(loss_storage_dir)
        utils.create_dir(loss_storage_dir)

        self.save_plot: bool = save_plot

        if self.save_plot:
            sns.set_style("white")
            sns.color_palette("bright")

    def on_evaluate(self,  # type: ignore
                    training_state: TrainingState,
                    training_args: TrainingArgs,
                    model: nn.Module,
                    **kwargs,
                   ) -> None:
        """
        Stores loss history on train and validation to the json file. Optionally this method can draw a plots
        with the loss history.

        Args:
            training_state (TrainingState): training state. It stores loss history, current epoch number and etc.
            trainihg_args (TrainingArgs): training arg that was created by the factory.
            model (nn.Module): model you want to evaluate.
        """
        with open(self.loss_storage_dir.joinpath("train_loss_history.json"), "w") as f:
            json.dump(training_state.train_loss_history, f)

        with open(self.loss_storage_dir.joinpath("eval_loss_history.json"), "w") as f:
            json.dump(training_state.eval_loss_history, f)

        if self.save_plot:
            sns.set()

            loss_train_history = list(training_state.train_loss_history.values())
            loss_val_history = list(training_state.eval_loss_history.values())

            epoch_history = list(training_state.train_loss_history.keys())

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax.tick_params(which="major", length=7)
            ax.tick_params(which="minor", length=4, color="r")

            ax.grid(which="minor", alpha=0.2)
            ax.grid(which="major", alpha=0.5)

            ax.plot(epoch_history, loss_train_history, label="Training")
            ax.plot(epoch_history, loss_val_history, label="Validation")

            plt.xticks(rotation=45)

            plt.title("Loss during trainig")
            plt.xlabel("Number of epochs")
            plt.legend()

            plt.savefig(self.loss_storage_dir.joinpath("loss.png"))


class CheckpointCallback(CustomCallback):
    def __init__(self, checkpoint_storage_dir: str | pathlib.Path):
        super().__init__()

        self.checkpoint_storage_dir: pathlib.Path = pathlib.Path(checkpoint_storage_dir)
        utils.create_dir(checkpoint_storage_dir)

    def on_save(self,  # type: ignore
                training_state: TrainingState,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                **kwargs,
               ) -> None:
        """
        Saves given model and optimizer at the end of every epoch.

        Args:
            training_state (TrainingState): training state. It stores loss history, current epoch number and etc.
            model (torch.nn.Module): model you want to save.
            optimizer (torch.optim.Optimizer): optimiezer you want to save.
        """
        # see https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        checkpoint = {
            "epoch": training_state.epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(checkpoint, self.checkpoint_storage_dir.joinpath("checkpoint.pth"))


class SaveBestModelCallback(CustomCallback):
    def __init__(self, checkpoint_storage_dir: str | pathlib.Path):
        super().__init__()

        self.checkpoint_storage_dir: pathlib.Path = pathlib.Path(checkpoint_storage_dir)
        utils.create_dir(checkpoint_storage_dir)

    def on_save(self,  # type: ignore
                training_state: TrainingState,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                **kwargs,
               ) -> None:
        """
        Saves given model in the case its eval loss if lesser than the stored one.

        Args:
            training_state (TrainingState): training state. It stores loss history, current epoch number and etc.
            model (torch.nn.Module): model you want to save.
            optimizer (torch.optim.Optimizer): optimiezer you want to save. Will be ignored.
        """
        eval_loss_last_value: float = list(training_state.eval_loss_history.values())[-1]

        if training_state.eval_loss_best_value > eval_loss_last_value:
            training_state.eval_loss_best_value = eval_loss_last_value

            checkpoint = {
                "model": model.state_dict(),
            }

            torch.save(checkpoint, self.checkpoint_storage_dir.joinpath("checkpoint.pth"))
