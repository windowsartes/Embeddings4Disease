import json
import pathlib
import warnings

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
try:
    import wandb  # type: ignore
except ImportError:
    warnings.warn(ImportWarning("wandb isn't installed so it won't be used."))
    wandb_installed: bool = False
else:
    wandb_installed = True

from embeddings4disease.data.datasets import CustomLineByLineDataset
from embeddings4disease.data.collators import MaskingCollator
from embeddings4disease.metrics.backbone_metrics import MLMMetricComputer
from embeddings4disease.utils import utils


class MetricComputerCallback(TrainerCallback):
    """
    This callback is used during training so we can evaluate model times after times.
    Using MaskingCollator it [MASK]s the last token in sequence and then asks model to fill the [MASK].
    Model will produce top_k best predictions and by their value will be validated using metrics you've
    specified in the 'use_metrics' argument. Please note that there can be only metrics  MetricComputer
    can work with.

    As input argumnets this class receives:

    Args:
        path_to_data (str): path to data on which you want to validate your model. It must by in line-by-line
            format: one transaction of icd10 codes on one line.
        metrics_storage_dir (str): path to directory where metrics values will be logged.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): tokenizer which will be used to tokenize tha data inside dataloader.
        use_metrics (dict[str, bool]): indicated which metrics you want to use for validation.
        device (torch.device): by design we will validate you model using cpu so after the inference the model
            will be return to the given device so it the model can continue the training.
        period (int, optional): all the metrics will be computed only once a {period} epochs. Defaults to 10.
        top_k (int, optional): number of prediction for every [MASK] your model will produce. Defaults to 10.
        batch_size (int, optional): batch size for dataloader. Defaults to 1024.
        seq_len (int, optional): maximum length in tokens one input sequence can have. Defaults to 24.
        use_wandb (bool, optional): allow wandb logging or not. Wandb module must be installed inside your venv.
        save_plot (bool, optional): should metrics and loss history be saved as a plot during training/validation.
    """

    def __init__(
        self,
        path_to_data: str | pathlib.Path,
        metrics_storage_dir: str | pathlib.Path,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        use_metrics: dict[str, bool],
        device: torch.device,
        period: int = 10,
        top_k: int = 10,
        batch_size: int = 1024,
        seq_len: int = 24,
        use_wandb: bool = False,
        save_plot: bool = True,
    ):
        super().__init__()

        self.metrics_storage_dir: pathlib.Path = pathlib.Path(metrics_storage_dir)
        utils.create_dir(self.metrics_storage_dir)

        self.period: int = period

        self.device: torch.device = device

        self.use_metrics: dict[str, bool] = use_metrics
        self.use_wandb: bool = use_wandb
        self.save_plot: bool = save_plot

        if self.save_plot:
            sns.set_style("white")
            sns.set_palette("bright")

        for metric, usage in self.use_metrics.items():
            if usage:
                self.__initialize_metric_storage(metric)

        dataset: Dataset = CustomLineByLineDataset(path_to_data)

        dataloader: DataLoader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=MaskingCollator(tokenizer, seq_len),
        )

        self.metric_computer: MLMMetricComputer = MLMMetricComputer(
            tokenizer, top_k, dataloader
        )

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

    def on_evaluate(  # type: ignore
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """
        This method will be called by Trainer on evaluation phase.
        Using MetricComputer it will compute metric you've selected in use_metrics argument and log them.

        Args:
            args (TrainingArguments): Trainer's TrainingArguments; see docs for more details.
            state (TrainerState): Trainer's TrainerState; see docs for more details.
            control (TrainerControl): Trainer's TrainerControl; see docs for more details.

        Returns:
            TrainerControl: Trainer's TrainerControl.
        """
        if (state.epoch - 1) % self.period == 0:
            metrics: dict[str, float] = self.metric_computer.get_metrics_value(
                kwargs["model"].to("cpu"), self.use_metrics
            )

            kwargs["model"].to(self.device)

            for metric_name, usage in self.use_metrics.items():
                if usage:
                    logs = self.__load_logs(metric_name)
                    logs[state.epoch] = metrics[metric_name]

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

                        plt.savefig(self.metrics_storage_dir.joinpath(
                            pathlib.Path(metric_name).joinpath(f"{metric_name}.png")
                        ))

            if wandb_installed and self.use_wandb:
                wandb.log({f"eval/{metric_name}": metrics[metric_name] for metric_name, usage in self.use_metrics.items() if usage})

        return control

class SaveLossHistoryCallback(TrainerCallback):
    """
    This callback is used to dump loss history to given file.

    Args:
        loss_storage_dir (str | pathlib.Path): directory where you want to store loss history.
    """

    def __init__(self, loss_storage_dir: str | pathlib.Path, save_plot: bool):
        super().__init__()

        self.loss_storage_dir: pathlib.Path = pathlib.Path(loss_storage_dir)
        utils.create_dir(loss_storage_dir)

        self.save_plot: bool = save_plot

        if self.save_plot:
            sns.set_style("white")
            sns.set_palette("bright")

    def on_evaluate(  # type: ignore
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """
        This method will be called on evaluation phase so we can store loss history in the given file.

        Args:
            args (TrainingArguments): Trainer's TrainingArguments; see docs for more details.
            state (TrainerState): Trainer's TrainerState; see docs for more details.
            control (TrainerControl): Trainer's TrainerControl; see docs for more details.

        Returns:
            TrainerControl: Trainer's TrainerControl.
        """
        history = state.log_history

        with open(self.loss_storage_dir.joinpath("loss_history.json"), "w") as f:
            json.dump(history, f)

        if self.save_plot:
            sns.set()

            loss_train_history = [record["loss"] for record in history if "loss" in record]
            loss_val_history = [
                record["eval_loss"] for record in history if "eval_loss" in record
            ]
            epoch_history = [int(record["epoch"]) for record in history if "loss" in record]

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

        return control
