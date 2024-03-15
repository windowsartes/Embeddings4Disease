import os
import pathlib
import typing as tp
from abc import ABC, abstractmethod
from math import ceil

import torch
import transformers
import yaml
from transformers import (
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TrainerCallback,
    TrainingArguments,
)

from callbacks import callbacks
from utils import utils


#dir_path: str = os.path.dirname(os.path.realpath(__file__))
#root_path: pathlib.Path = pathlib.Path(dir_path).parents[1]

working_dir = pathlib.Path(os.getcwd())

print(working_dir)
print("---------------------")


class ArchitectureFactory(ABC):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    @abstractmethod
    def construct(self) -> tuple[
        PreTrainedModel,
        PreTrainedTokenizer | PreTrainedTokenizerFast,
        TrainingArguments,
        list[TrainerCallback],
    ]:
        pass

    def _create_collator_and_datasets(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    ) -> tuple[
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
    ]:
        seq_len: int = self.config["hyperparameters"]["seq_len"]
        batch_size: int = self.config["hyperparameters"]["batch_size"]

        dataset_train: LineByLineTextDataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=working_dir.joinpath(self.config["training"]["path_to_data"]),
            block_size=seq_len + 2,
        )

        dataset_eval: LineByLineTextDataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=working_dir.joinpath(self.config["validation"]["path_to_data"]),
            block_size=seq_len + 2,
        )

        data_collator: DataCollatorForLanguageModeling = (
            DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15,
                return_tensors="pt",
            )
        )

        return (
            data_collator,
            dataset_train,
            dataset_eval,
        )

    def _create_callbacks(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    ) -> list[TrainerCallback]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        device: torch.device = torch.device(
            self.config["training"]["device"] if torch.cuda.is_available() else "cpu"
        )

        used_callbacks: list[transformers.TrainerCallback] = []

        compute_metrics: bool = False
        for value in self.config["validation"]["metrics"].values():
            if value:
                compute_metrics = value
                break

        if compute_metrics:
            used_callbacks.append(
                callbacks.MetricComputerValidationCallback(
                    path_to_data=working_dir.joinpath(
                        self.config["validation"]["path_to_data"]
                    ),
                    path_to_storage=storage_path.joinpath("metrics"),
                    tokenizer=tokenizer,
                    use_metrics=self.config["validation"]["metrics"],
                    device=device,
                    period=self.config["validation"]["period"],
                    top_k=self.config["validation"]["top_k"],
                    batch_size=batch_size,
                    seq_len=seq_len,
                ),
            )

        if self.config["validation"]["save_graphs"]:
            used_callbacks.append(
                callbacks.SaveGraphsCallback(storage_path=storage_path)
            )

        if self.config["save_trained_model"]:
            used_callbacks.append(
                callbacks.SaveLossHistoryCallback(storage_path=storage_path)
            )

        return used_callbacks


C = tp.TypeVar("C", bound=ArchitectureFactory)

NAME_TO_CLASS: dict[str, tp.Type[C]] = {}  # название не лучшее, просто для примера


def architecture(cls: tp.Type[C]) -> tp.Type[C]:
    NAME_TO_CLASS[cls.__name__[:-7]] = cls
    return cls


@architecture
class BERTFactory(ArchitectureFactory):
    def construct(
        self,
    ) -> tuple[
        PreTrainedModel,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
        list[TrainerCallback],
    ]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        for name in ["checkpoint", "metrics", "graphs", "loss"]:
            utils.create_dir(storage_path.joinpath(name))

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        n_epochs: int = self.config["training"]["n_epochs"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        if self.config["tokenizer"]["use_pretrained"]:
            tokenizer: transformers.BertTokenizer = (
                transformers.BertTokenizer.from_pretrained(
                    "windowsartes/bert_tokenizer"
                )
            )
        else:
            pass

        if self.config["model"]["use_pretrained"]:
            model: transformers.BertForMaskedLM = (
                transformers.BertForMaskedLM.from_pretrained("windowsartes/bert")
            )
        else:
            config: transformers.BertConfig = transformers.BertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model: transformers.BertForMaskedLM = transformers.BertForMaskedLM(
                config=config
            )

        data_collator, dataset_train, dataset_eval = self._create_collator_and_datasets(
            tokenizer
        )

        n_warmup_steps: int = ceil(len(dataset_train) / batch_size) * n_warmup_epochs

        training_args = transformers.TrainingArguments(
            output_dir=storage_path.joinpath("checkpoint"),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            prediction_loss_only=True,
            warmup_steps=n_warmup_steps,
            report_to="none",
            **self.config["training"]["optimizer_parameters"],
        )

        used_callbacks = self._create_callbacks(tokenizer)

        return (
            model,
            training_args,
            data_collator,
            dataset_train,
            dataset_eval,
            used_callbacks,
        )


@architecture
class ConvBERTFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    def construct(
        self,
    ) -> tuple[
        PreTrainedModel,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
        list[TrainerCallback],
    ]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        for name in ["checkpoint", "metrics", "graphs", "loss"]:
            utils.create_dir(storage_path.joinpath(name))

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        n_epochs: int = self.config["training"]["n_epochs"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        if self.config["tokenizer"]["use_pretrained"]:
            tokenizer: transformers.ConvBertTokenizer = (
                transformers.ConvBertTokenizer.from_pretrained(
                    "windowsartes/convbert_tokenizer"
                )
            )
        else:
            pass

        if self.config["model"]["use_pretrained"]:
            model: transformers.ConvBertForMaskedLM = (
                transformers.ConvBertForMaskedLM.from_pretrained(
                    "windowsartes/convbert"
                )
            )
        else:
            config: transformers.ConvBertConfig = transformers.ConvBertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model: transformers.ConvBertForMaskedLM = transformers.ConvBertForMaskedLM(
                config=config
            )

        data_collator, dataset_train, dataset_eval = self._create_collator_and_datasets(
            tokenizer
        )

        n_warmup_steps: int = ceil(len(dataset_train) / batch_size) * n_warmup_epochs

        training_args = transformers.TrainingArguments(
            output_dir=working_dir.joinpath(self.config["training"]["checkpoint_dir"]),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            prediction_loss_only=True,
            warmup_steps=n_warmup_steps,
            report_to="none",
            **self.config["training"]["optimizer_parameters"],
        )

        used_callbacks = self._create_callbacks(tokenizer)

        return (
            model,
            training_args,
            data_collator,
            dataset_train,
            dataset_eval,
            used_callbacks,
        )


@architecture
class DeBERTaFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    def construct(
        self,
    ) -> tuple[
        PreTrainedModel,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
        list[TrainerCallback],
    ]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        for name in ["checkpoint", "metrics", "graphs", "loss"]:
            utils.create_dir(storage_path.joinpath(name))

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        n_epochs: int = self.config["training"]["n_epochs"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        if self.config["tokenizer"]["use_pretrained"]:
            tokenizer: transformers.DebertaTokenizerFast = (
                transformers.DebertaTokenizerFast.from_pretrained(
                    "windowsartes/deberta_tokenizer_fast"
                )
            )
        else:
            pass

        if self.config["model"]["use_pretrained"]:
            model: transformers.DebertaForMaskedLM = (
                transformers.DebertaForMaskedLM.from_pretrained("windowsartes/deberta")
            )
        else:
            config: transformers.DebertaConfig = transformers.DebertaConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model: transformers.DebertaForMaskedLM = transformers.DebertaForMaskedLM(
                config=config
            )

        data_collator, dataset_train, dataset_eval = self._create_collator_and_datasets(
            tokenizer
        )

        n_warmup_steps: int = ceil(len(dataset_train) / batch_size) * n_warmup_epochs

        training_args = transformers.TrainingArguments(
            output_dir=working_dir.joinpath(self.config["training"]["checkpoint_dir"]),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            prediction_loss_only=True,
            warmup_steps=n_warmup_steps,
            report_to="none",
            **self.config["training"]["optimizer_parameters"],
        )

        used_callbacks = self._create_callbacks(tokenizer)

        return (
            model,
            training_args,
            data_collator,
            dataset_train,
            dataset_eval,
            used_callbacks,
        )


@architecture
class FNetFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    def construct(
        self,
    ) -> tuple[
        PreTrainedModel,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
        list[TrainerCallback],
    ]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        for name in ["checkpoint", "metrics", "graphs", "loss"]:
            utils.create_dir(storage_path.joinpath(name))

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        n_epochs: int = self.config["training"]["n_epochs"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        if self.config["tokenizer"]["use_pretrained"]:
            tokenizer: transformers.FNetTokenizerFast = (
                transformers.FNetTokenizerFast.from_pretrained(
                    "windowsartes/fnet_tokenizer_fast"
                )
            )
        else:
            pass

        if self.config["model"]["use_pretrained"]:
            model: transformers.FNetForMaskedLM = (
                transformers.FNetForMaskedLM.from_pretrained("windowsartes/fnet")
            )
        else:
            config: transformers.FNetConfig = transformers.FNetConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model: transformers.FNetForMaskedLM = transformers.FNetForMaskedLM(
                config=config
            )

        data_collator, dataset_train, dataset_eval = self._create_collator_and_datasets(
            tokenizer
        )

        n_warmup_steps: int = ceil(len(dataset_train) / batch_size) * n_warmup_epochs

        training_args = transformers.TrainingArguments(
            output_dir=working_dir.joinpath(self.config["training"]["checkpoint_dir"]),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            prediction_loss_only=True,
            warmup_steps=n_warmup_steps,
            report_to="none",
            **self.config["training"]["optimizer_parameters"],
        )

        used_callbacks = self._create_callbacks(tokenizer)

        return (
            model,
            training_args,
            data_collator,
            dataset_train,
            dataset_eval,
            used_callbacks,
        )


@architecture
class FunnelTransformerFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    def construct(
        self,
    ) -> tuple[
        PreTrainedModel,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
        list[TrainerCallback],
    ]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        for name in ["checkpoint", "metrics", "graphs", "loss"]:
            utils.create_dir(storage_path.joinpath(name))

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        n_epochs: int = self.config["training"]["n_epochs"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        if self.config["tokenizer"]["use_pretrained"]:
            tokenizer: transformers.FunnelTokenizer = (
                transformers.FunnelTokenizer.from_pretrained(
                    "windowsartes/funnel_tokenizer"
                )
            )
        else:
            pass

        if self.config["model"]["use_pretrained"]:
            model: transformers.FunnelForMaskedLM = (
                transformers.FunnelForMaskedLM.from_pretrained("windowsartes/funnel")
            )
        else:
            config: transformers.FunnelConfig = transformers.FunnelConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                **self.config["model"]["config"],
            )
            model: transformers.FunnelForMaskedLM = transformers.FunnelForMaskedLM(
                config=config
            )

        data_collator, dataset_train, dataset_eval = self._create_collator_and_datasets(
            tokenizer
        )

        n_warmup_steps: int = ceil(len(dataset_train) / batch_size) * n_warmup_epochs

        training_args = transformers.TrainingArguments(
            output_dir=working_dir.joinpath(self.config["training"]["checkpoint_dir"]),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            prediction_loss_only=True,
            warmup_steps=n_warmup_steps,
            report_to="none",
            **self.config["training"]["optimizer_parameters"],
        )

        used_callbacks = self._create_callbacks(tokenizer)

        return (
            model,
            training_args,
            data_collator,
            dataset_train,
            dataset_eval,
            used_callbacks,
        )


@architecture
class MobileBERTFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    def construct(
        self,
    ) -> tuple[
        PreTrainedModel,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
        list[TrainerCallback],
    ]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        for name in ["checkpoint", "metrics", "graphs", "loss"]:
            utils.create_dir(storage_path.joinpath(name))

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        n_epochs: int = self.config["training"]["n_epochs"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        if self.config["tokenizer"]["use_pretrained"]:
            tokenizer: transformers.MobileBertTokenizer = (
                transformers.MobileBertTokenizer.from_pretrained(
                    "windowsartes/mobilebert_tokenizer"
                )
            )
        else:
            pass

        if self.config["model"]["use_pretrained"]:
            model: transformers.MobileBertForMaskedLM = (
                transformers.MobileBertForMaskedLM.from_pretrained(
                    "windowsartes/mobilebert"
                )
            )
        else:
            config: transformers.MobileBertConfig = transformers.MobileBertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model: transformers.MobileBertForMaskedLM = (
                transformers.MobileBertForMaskedLM(config=config)
            )

        data_collator, dataset_train, dataset_eval = self._create_collator_and_datasets(
            tokenizer
        )

        n_warmup_steps: int = ceil(len(dataset_train) / batch_size) * n_warmup_epochs

        training_args = transformers.TrainingArguments(
            output_dir=working_dir.joinpath(self.config["training"]["checkpoint_dir"]),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            prediction_loss_only=True,
            warmup_steps=n_warmup_steps,
            report_to="none",
            **self.config["training"]["optimizer_parameters"],
        )

        used_callbacks = self._create_callbacks(tokenizer)

        return (
            model,
            training_args,
            data_collator,
            dataset_train,
            dataset_eval,
            used_callbacks,
        )


@architecture
class RoBERTaFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    def construct(
        self,
    ) -> tuple[
        PreTrainedModel,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
        list[TrainerCallback],
    ]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        for name in ["checkpoint", "metrics", "graphs", "loss"]:
            utils.create_dir(storage_path.joinpath(name))

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        n_epochs: int = self.config["training"]["n_epochs"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        if self.config["tokenizer"]["use_pretrained"]:
            tokenizer: transformers.RobertaTokenizerFast = (
                transformers.RobertaTokenizerFast.from_pretrained(
                    "windowsartes/roberta_tokenizer_fast"
                )
            )
        else:
            pass

        if self.config["model"]["use_pretrained"]:
            model: transformers.RobertaForMaskedLM = (
                transformers.RobertaForMaskedLM.from_pretrained("windowsartes/roberta")
            )
        else:
            config: transformers.RobertaConfig = transformers.RobertaConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 4,
                **self.config["model"]["config"],
            )
            model: transformers.RobertaForMaskedLM = transformers.RobertaForMaskedLM(
                config=config
            )

        data_collator, dataset_train, dataset_eval = self._create_collator_and_datasets(
            tokenizer
        )

        n_warmup_steps: int = ceil(len(dataset_train) / batch_size) * n_warmup_epochs

        training_args = transformers.TrainingArguments(
            output_dir=working_dir.joinpath(self.config["training"]["checkpoint_dir"]),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            prediction_loss_only=True,
            warmup_steps=n_warmup_steps,
            report_to="none",
            **self.config["training"]["optimizer_parameters"],
        )

        used_callbacks = self._create_callbacks(tokenizer)

        return (
            model,
            training_args,
            data_collator,
            dataset_train,
            dataset_eval,
            used_callbacks,
        )


@architecture
class RoFormerFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    def construct(
        self,
    ) -> tuple[
        PreTrainedModel,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
        list[TrainerCallback],
    ]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        for name in ["checkpoint", "metrics", "graphs", "loss"]:
            utils.create_dir(storage_path.joinpath(name))

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        n_epochs: int = self.config["training"]["n_epochs"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        if self.config["tokenizer"]["use_pretrained"]:
            tokenizer: transformers.RoFormerTokenizer = (
                transformers.RoFormerTokenizer.from_pretrained(
                    "windowsartes/roformer_tokenizer"
                )
            )
        else:
            pass

        if self.config["model"]["use_pretrained"]:
            model: transformers.RoFormerForMaskedLM = (
                transformers.RoFormerForMaskedLM.from_pretrained(
                    "windowsartes/roformer"
                )
            )
        else:
            config: transformers.RoFormerConfig = transformers.RoFormerConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 4,
                **self.config["model"]["config"],
            )
            model: transformers.RoFormerForMaskedLM = transformers.RoFormerForMaskedLM(
                config=config
            )

        data_collator, dataset_train, dataset_eval = self._create_collator_and_datasets(
            tokenizer
        )

        n_warmup_steps: int = ceil(len(dataset_train) / batch_size) * n_warmup_epochs

        training_args = transformers.TrainingArguments(
            output_dir=working_dir.joinpath(self.config["training"]["checkpoint_dir"]),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            prediction_loss_only=True,
            warmup_steps=n_warmup_steps,
            report_to="none",
            **self.config["training"]["optimizer_parameters"],
        )

        used_callbacks = self._create_callbacks(tokenizer)

        return (
            model,
            training_args,
            data_collator,
            dataset_train,
            dataset_eval,
            used_callbacks,
        )


@architecture
class XLMRoBERTaFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    def construct(
        self,
    ) -> tuple[
        PreTrainedModel,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        LineByLineTextDataset,
        LineByLineTextDataset,
        list[TrainerCallback],
    ]:
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        for name in ["checkpoint", "metrics", "graphs", "loss"]:
            utils.create_dir(storage_path.joinpath(name))

        batch_size: int = self.config["hyperparameters"]["batch_size"]
        seq_len: int = self.config["hyperparameters"]["seq_len"]

        n_epochs: int = self.config["training"]["n_epochs"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        if self.config["tokenizer"]["use_pretrained"]:
            tokenizer: transformers.XLMRobertaTokenizerFast = (
                transformers.XLMRobertaTokenizerFast.from_pretrained(
                    "windowsartes/xlmroberta_tokenizer_fast"
                )
            )
        else:
            pass

        if self.config["model"]["use_pretrained"]:
            model: transformers.XLMRobertaForMaskedLM = (
                transformers.XLMRobertaForMaskedLM.from_pretrained(
                    "windowsartes/xlmroberta"
                )
            )
        else:
            config: transformers.XLMRobertaConfig = transformers.XLMRobertaConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 4,
                **self.config["model"]["config"],
            )
            model: transformers.XLMRobertaForMaskedLM = (
                transformers.XLMRobertaForMaskedLM(config=config)
            )

        data_collator, dataset_train, dataset_eval = self._create_collator_and_datasets(
            tokenizer
        )

        n_warmup_steps: int = ceil(len(dataset_train) / batch_size) * n_warmup_epochs

        training_args = transformers.TrainingArguments(
            output_dir=working_dir.joinpath(self.config["training"]["checkpoint_dir"]),
            overwrite_output_dir=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            prediction_loss_only=True,
            warmup_steps=n_warmup_steps,
            report_to="none",
            **self.config["training"]["optimizer_parameters"],
        )

        used_callbacks = self._create_callbacks(tokenizer)

        return (
            model,
            training_args,
            data_collator,
            dataset_train,
            dataset_eval,
            used_callbacks,
        )
