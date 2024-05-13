import os
import pathlib
import typing as tp
from abc import ABC, abstractmethod
from datetime import datetime

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM

from embeddings4disease.callbacks import custom_callbacks, hf_callbacks
from embeddings4disease.head.architectures import multilabel_head
from embeddings4disease.data.datasets import MultiLabelHeadDataset, EncoderDecoderDataset
from embeddings4disease.data.collators import MultiLabelHeadCollator
from embeddings4disease.metrics import multilabel_head_metrics
from embeddings4disease.trainer.training_args import TrainingArgs
from embeddings4disease.utils import utils


class HeadFactory(ABC):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    @abstractmethod
    def initialize(self) -> None:
        """
        Additional initialization for your factory. Usually this method is used during the traing and it will be ignored
        during the model's final validation.
        """
        pass

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        """
        Creates the model you want to train.

        Returns:
            torch.nn.Module: model you will use later.
        """
        pass

    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        Sinse we use pretrained backbone here we'll load pretrained tokenizer.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: pretrained tokenizer.
        """
        pass

    @abstractmethod
    def create_training_args(self) -> TrainingArgs | transformers.TrainingArguments:
        """
        Training args for the Trainer.

        Returns:
            TrainingArgs: Trainer's training arguments.
        """
        pass


    @abstractmethod
    def create_callbacks(self) -> list[transformers.TrainerCallback] | list[custom_callbacks.CustomCallback]:
        pass

    @abstractmethod
    def create_dataset(self, mode: str) -> Dataset:
        pass

    @abstractmethod
    def create_collator(self) -> tp.Callable:
        pass

    @abstractmethod
    def create_metric_computer(self) -> tuple[multilabel_head_metrics.MetricComputerInterface, dict[str, bool]]:
        pass

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


class CustomHeadFactory(HeadFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    @abstractmethod
    def create_callbacks(self) -> list[custom_callbacks.CustomCallback]:
        """
        Creates a list of callbacks that will be used during trainig.

        Returns:
            list[custom_callbacks.CustomCallback]: list of callback that will be used later.
        """
        pass

    @abstractmethod
    def create_training_args(self) -> TrainingArgs:
        """
        Training args for the Trainer.

        Returns:
            TrainingArgs: Trainer's training arguments.
        """
        pass


class HuggingFaceHeadFactory(HeadFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    @abstractmethod
    def create_collator(self) -> transformers.DataCollatorForSeq2Seq:
        pass

    @abstractmethod
    def create_dataset(self, mode: str) -> EncoderDecoderDataset:
        pass

    @abstractmethod
    def create_training_args(self) -> transformers.TrainingArguments:
        pass

    @abstractmethod
    def create_callbacks(self) -> list[transformers.TrainerCallback]:
        pass


HEAD_REGISTER: dict[str, tp.Type[HeadFactory]] = {}


def head(cls: tp.Type[HeadFactory]) -> tp.Type[HeadFactory]:
    HEAD_REGISTER[cls.__name__[:-11]] = cls
    return cls


@head
class MultiLabelHeadFactory(CustomHeadFactory):
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

        model = multilabel_head.MultiLabelHead(
            backbone,
            tokenizer.vocab_size,
            **self.config["model"]["head"],
        )

        if self.config["model"]["from_pretrained"]:
            model.load_state_dict(torch.load(os.path.abspath(self.config["model"]["path_to_pretrained_model"])))

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self.config["tokenizer"]["from_huggingface"]:
            return AutoTokenizer.from_pretrained(
                self.config["tokenizer"]["path_to_saved_tokenizer"]
            )

        return AutoTokenizer.from_pretrained(
            os.path.abspath(self.config["tokenizer"]["path_to_saved_tokenizer"])
        )

    def create_dataset(self, mode: str) -> MultiLabelHeadDataset:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        dataset: MultiLabelHeadDataset = MultiLabelHeadDataset(
            os.path.abspath(self.config[mode]["path_to_data"]), tokenizer
        )

        return dataset

    def create_collator(self) -> MultiLabelHeadCollator:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        collator: MultiLabelHeadCollator = MultiLabelHeadCollator(
            tokenizer, self.config["hyperparameters"]["seq_len"]
        )

        return collator

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
                    device=device,
                    period=self.config["validation"]["period"],
                    threshold=self.config["validation"]["threshold"],
                    save_plot=self.config["validation"]["save_graphs"],
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
            batch_size=self.config["hyperparameters"]["batch_size"],
            **self.config["training"]["optimizer_parameters"],
        )

    def create_metric_computer(self) -> tuple[multilabel_head_metrics.MultiLabelHeadMetricComputer, dict[str, bool]]:
        dataset: MultiLabelHeadDataset = self.create_dataset("validation")

        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=self.config["hyperparameters"]["batch_size"],
            collate_fn=self.create_collator(),
        )

        metric_computer: multilabel_head_metrics.MultiLabelHeadMetricComputer = \
            multilabel_head_metrics.MultiLabelHeadMetricComputer(
                self.config["validation"]["threshold"],
                dataloader,
                torch.device("cpu"),
                self.config["validation"]["confidence_interval"],
                self.config["validation"]["interval_type"],
                self.config["validation"]["confidence_level"],
            )

        return (metric_computer, self.config["validation"]["metrics"])


@head
class EncoderDecoderHeadFactory(HuggingFaceHeadFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def initialize(self) -> None:
        self._create_storage()

    def load_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self.config["tokenizer"]["from_huggingface"]:
            return AutoTokenizer.from_pretrained(
                self.config["tokenizer"]["path_to_saved_tokenizer"]
            )

        return AutoTokenizer.from_pretrained(
            os.path.abspath(self.config["tokenizer"]["path_to_saved_tokenizer"])
        )

    def create_model(self) -> transformers.EncoderDecoderModel:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        if self.config["model"]["from_encoder_decoder"]:
            if self.config["model"]["encode"]["from_huggingface"]:
                path_to_saved_encoder: str = self.config["model"]["encode"]["path_to_saved_model"]
            else:
                path_to_saved_encoder = os.path.abspath(self.config["model"]["encode"]["path_to_saved_model"])

            if self.config["model"]["decoder"]["from_huggingface"]:
                path_to_saved_decoder: str = self.config["model"]["decoder"]["path_to_saved_model"]
            else:
                path_to_saved_decoder = os.path.abspath(self.config["model"]["decoder"]["path_to_saved_model"])

            self._model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
                path_to_saved_encoder,
                path_to_saved_decoder,
            )
        else:
            if self.config["model"]["saved_model"]["from_huggingface"]:
                path_to_saved_model: str = self.config["model"]["saved_model"]["path_to_saved_model"]
            else:
                path_to_saved_model = os.path.abspath(self.config["model"]["saved_model"]["path_to_saved_model"])

            self._model = transformers.EncoderDecoderModel.from_pretrained(path_to_saved_model)

        self._model.config.decoder.decoder_start_token_id = tokenizer.cls_token_id
        self._model.config.decoder.pad_token_id = tokenizer.pad_token_id
        self._model.config.decoder.bos_token_id = tokenizer.cls_token_id

        self._model.config.encoder.decoder_start_token_id = tokenizer.cls_token_id
        self._model.config.encoder.pad_token_id = tokenizer.pad_token_id
        self._model.config.encoder.bos_token_id = tokenizer.cls_token_id

        self._model.config.decoder_start_token_id = tokenizer.cls_token_id
        self._model.config.pad_token_id = tokenizer.pad_token_id
        self._model.config.bos_token_id = tokenizer.cls_token_id

        self._model.generation_config.decoder_start_token_id = tokenizer.cls_token_id

        return self._model

    def create_collator(self) -> transformers.DataCollatorForSeq2Seq:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()
        max_length: int = self.config["hyperparameters"]["seq_len"]

        collator: transformers.DataCollatorForSeq2Seq = transformers.DataCollatorForSeq2Seq(
            tokenizer,
            model=self._model,
            padding="longest",
            max_length=max_length,
        )

        return collator

    def create_dataset(self, mode: str) -> EncoderDecoderDataset:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.load_tokenizer()
        )

        dataset: EncoderDecoderDataset = EncoderDecoderDataset(
            path=os.path.abspath(self.config[mode]["path_to_data"]),
            tokenizer=tokenizer,
            max_length=self.config["hyperparameters"]["seq_len"],
        )

        return dataset

    def create_training_args(self) -> transformers.Seq2SeqTrainingArguments:
        checkpoint_path: pathlib.Path = self.storage_path.joinpath("checkpoint")

        utils.create_dir(checkpoint_path)
        utils.delete_files(checkpoint_path)

        args = transformers.Seq2SeqTrainingArguments(
            output_dir=str(checkpoint_path),
            overwrite_output_dir=True,
            num_train_epochs=self.config["training"]["n_epochs"],
            per_device_train_batch_size=self.config["hyperparameters"]["batch_size"],
            per_device_eval_batch_size=self.config["hyperparameters"]["batch_size"]*2,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            predict_with_generate=True,
            fp16=True,
            prediction_loss_only=True,
        )

        return args

    def create_callbacks(self) -> list[transformers.TrainerCallback]:
        used_callbacks = []

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
                hf_callbacks.EncoderDecoderMetricComputerCallback(
                    path_to_data=self.config["validation"]["path_to_data"],
                    metrics_storage_dir=self.storage_path.joinpath("metrics"),
                    tokenizer=self.load_tokenizer(),
                    use_metrics=self.config["validation"]["metrics"],
                    device=device,
                    period=self.config["validation"]["period"],
                    batch_size=self.config["hyperparameters"]["batch_size"],
                    seq_len=self.config["hyperparameters"]["seq_len"],
                    use_wandb=False,
                    save_plot=self.config["validation"]["save_graphs"],
                )
            )

        used_callbacks.append(
            hf_callbacks.SaveLossHistoryCallback(
                loss_storage_dir=self.storage_path.joinpath("loss"),
                save_plot=self.config["validation"]["save_graphs"],
            )
        )

        return used_callbacks

    def create_metric_computer(self) -> tuple[multilabel_head_metrics.EncoderDecoderHeadMetricComputer, dict[str, bool]]:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        dataset: EncoderDecoderDataset = self.create_dataset("validation")

        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=self.config["hyperparameters"]["batch_size"],
            collate_fn=self.create_collator(),
        )

        metric_computer: multilabel_head_metrics.EncoderDecoderHeadMetricComputer = \
            multilabel_head_metrics.EncoderDecoderHeadMetricComputer(
                dataloader,
                torch.device("cpu"),
                tokenizer,
                self.config["hyperparameters"]["seq_len"],
            )

        return (metric_computer, self.config["validation"]["metrics"])
