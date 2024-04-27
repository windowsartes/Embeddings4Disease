import os
import typing as tp
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM

from embeddings4disease.head.architectures import multilabel_head
from embeddings4disease.data.datasets import MultiLabelHeadDataset
from embeddings4disease.data.collators import MultiLabelHeadCollator
from embeddings4disease.trainer import TrainingArgs


class HeadFactory(ABC):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

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
    def create_callbacks(self) -> list[tp.Any]:  # just for now
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

    def create_callbacks(self) -> list[tp.Any]:
        return []

    def create_training_args(self) -> TrainingArgs:
        return TrainingArgs(
            mode=self.config["model"]["head"]["mode"],
            n_epochs=self.config["training"]["n_epochs"],
            n_warmup_epochs=self.config["training"]["n_warmup_epochs"],
            device=torch.device(self.config["training"]["device"]),
            criterion=torch.nn.BCEWithLogitsLoss,
            **self.config["training"]["optimizer_parameters"],
        )
