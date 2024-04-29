import os
import pathlib
import typing as tp
from abc import ABC, abstractmethod

import torch
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM

from embeddings4disease.callbacks import custom_callbacks
from embeddings4disease.head.architectures import multilabel_head
from embeddings4disease.data.datasets import MultiLabelHeadDataset
from embeddings4disease.data.collators import MultiLabelHeadCollator
from embeddings4disease.trainer import TrainingArgs
from embeddings4disease.utils import utils


class HeadFactory(ABC):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        pass

    @abstractmethod
    def create_dataloader(self, mode: str) -> DataLoader:
        pass

    @abstractmethod
    def create_callbacks(self) -> list[custom_callbacks.CustomCallback]:
        pass

    @abstractmethod
    def create_training_args(self) -> TrainingArgs:
        pass


HEAD_REGISTER: dict[str, tp.Type[HeadFactory]] = {}


def head(cls: tp.Type[HeadFactory]) -> tp.Type[HeadFactory]:
    HEAD_REGISTER[cls.__name__[:-7]] = cls
    return cls


@head
class MultiLabelHeadFactory(HeadFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def initialize(self) -> None:
        self._create_storage()

    def create_model(self) -> multilabel_head.MultiLabelHead:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        if self.config["model"]["backbone"]["from_huggingface"]:
            backbone: AutoModelForMaskedLM = AutoModelForMaskedLM.from_pretrained(
                self.config["model"]["backbone"]["path_to_saved_model"]
            )
        else:
            backbone = AutoModelForMaskedLM.from_pretrained(
                os.path.abspath(self.config["model"]["backbone"]["path_to_saved_model"])
            )

        model = multilabel_head.MultiLabelHead(backbone, tokenizer.vocab_size,
                                               **self.config["model"]["head"]
                                              )

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self.config["tokenizer"]["from_huggingface"]:
            return AutoTokenizer.from_pretrained(
                self.config["tokenizer"]["path_to_saved_tokenizer"]
            )

        return AutoTokenizer.from_pretrained(
            os.path.abspath(self.config["tokenizer"]["path_to_saved_tokenizer"])
        )

    def create_dataloader(self, mode: str) -> DataLoader:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        dataset: MultiLabelHeadDataset = MultiLabelHeadDataset(
            os.path.abspath(self.config[mode]["path_to_data"]), tokenizer
        )
        collate_fn: MultiLabelHeadCollator = MultiLabelHeadCollator(
            tokenizer, self.config["hyperparameters"]["seq_len"]
        )

        return DataLoader(
            dataset,
            batch_size=self.config["hyperparameters"]["batch_size"],
            collate_fn=collate_fn,
            # shuffle=True if mode=="training" else False,
            shuffle=True,
            drop_last=True,
        )

    def create_callbacks(self) -> list[custom_callbacks.CustomCallback]:
        used_callbacks: list[custom_callbacks.CustomCallback] = []

        compute_metrics: bool = False
        for value in self.config["validation"]["metrics"].values():
            if value:
                compute_metrics = value
                break

        if compute_metrics:
            device: torch.device = torch.device(
                self.config["training"]["device"]
                if torch.cuda.is_available()
                else "cpu"
            )

            used_callbacks.append(
                custom_callbacks.MetricComputerCallback(
                    metrics_storage_dir=self.storage_path.joinpath("metrics"),
                    use_metrics=self.config["validation"]["metrics"],
                    dataloader=self.create_dataloader("validation"),
                    device=device,
                    period=self.config["validation"]["period"],
                    threshold=self.config["validation"]["threshold"],
                    save_plot=True,
                )
            )

        used_callbacks.append(custom_callbacks.SaveLossHistoryCallback(self.storage_path.joinpath("loss"), True))

        used_callbacks.append(custom_callbacks.CheckpointCallback(self.storage_path.joinpath("checkpoint")))
        used_callbacks.append(custom_callbacks.SaveBestModelCallback(self.storage_path.joinpath("best_model")))

        return used_callbacks

    def create_training_args(self) -> TrainingArgs:
        return TrainingArgs(
            mode=self.config["model"]["head"]["mode"],
            n_epochs=self.config["training"]["n_epochs"],
            n_warmup_epochs=self.config["training"]["n_warmup_epochs"],
            device=torch.device(self.config["training"]["device"]),
            criterion=torch.nn.BCEWithLogitsLoss,
            **self.config["training"]["optimizer_parameters"],
        )

    def _create_storage(self) -> None:
        """
        This method is used to initialize storage dir in the case you need to store logs/graphs/etc somewhere.
        """
        working_dir: pathlib.Path = pathlib.Path(utils.get_cwd())

        now = datetime.now()
        data, time = now.strftime("%b-%d-%Y %H:%M").replace(":", "-").split()

        storage_path = working_dir.joinpath(self.config["model"]["type"]).joinpath(data).joinpath(time)
        utils.create_dir(storage_path)

        self.storage_path: pathlib.Path = storage_path
