import os
import pathlib
import shutil
import typing as tp
from abc import ABC, abstractmethod
from datetime import datetime
from math import ceil

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import GenerationConfig

from embeddings4disease.callbacks import custom_callbacks, hf_callbacks
from embeddings4disease.head.architectures import multilabel_head
from embeddings4disease.data.datasets import MultiLabelHeadDataset, EncoderDecoderDataset
from embeddings4disease.data.collators import MultiLabelHeadCollator
from embeddings4disease.metrics import head_metrics
from embeddings4disease.trainer.training_args import TrainingArgs
from embeddings4disease.utils import utils


class HeadFactory(ABC):
    def __init__(self, config: dict[str, tp.Any]):
        self._config: dict[str, tp.Any] = config

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
    def create_metric_computer(self) -> head_metrics.MetricComputerInterface:
        pass

    def _create_storage(self) -> None:
        """
        This method is used to initialize storage dir in the case you need to store logs/graphs/etc somewhere.
        """
        working_dir: pathlib.Path = pathlib.Path(utils.get_cwd())

        now = datetime.now()
        data, time = now.strftime("%b-%d-%Y %H:%M").replace(":", "-").split()

        storage_path = working_dir.joinpath(self._config["model"]["type"]).joinpath(data).joinpath(time)
        utils.create_dir(storage_path)

        self._storage_path: pathlib.Path = storage_path


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

    def set_warmup_epochs(
        self,
        training_args: transformers.TrainingArguments,
        dataset_train: EncoderDecoderDataset  # type: ignore
    ) -> None:
        batch_size: int = self._config["hyperparameters"]["batch_size"]
        n_warmup_epochs: int = self._config["training"]["n_warmup_epochs"]

        training_args.warmup_steps = (
            ceil(len(dataset_train) / batch_size) * n_warmup_epochs
        )


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

        if self._config["model"]["backbone"]["from_huggingface"]:
            backbone: AutoModelForMaskedLM = AutoModelForMaskedLM.from_pretrained(
                self._config["model"]["backbone"]["path_to_saved_model"]
            )
        else:
            backbone = AutoModelForMaskedLM.from_pretrained(
                os.path.abspath(self._config["model"]["backbone"]["path_to_saved_model"])
            )

        model = multilabel_head.MultiLabelHead(
            backbone,
            tokenizer.vocab_size,
            **self._config["model"]["head"],
        )

        if self._config["model"]["from_pretrained"]:
            model.load_state_dict(
                torch.load(
                    os.path.abspath(self._config["model"]["path_to_pretrained_model"]),
                    map_location=torch.device(self._config["training"]["device"]),
                )
            )

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self._config["tokenizer"]["from_huggingface"]:
            return AutoTokenizer.from_pretrained(
                self._config["tokenizer"]["path_to_saved_tokenizer"]
            )

        return AutoTokenizer.from_pretrained(
            os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
        )

    def create_dataset(self, mode: str) -> MultiLabelHeadDataset:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        dataset: MultiLabelHeadDataset = MultiLabelHeadDataset(
            os.path.abspath(self._config[mode]["path_to_data"]), tokenizer
        )

        return dataset

    def create_collator(self) -> MultiLabelHeadCollator:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        collator: MultiLabelHeadCollator = MultiLabelHeadCollator(
            tokenizer, self._config["hyperparameters"]["seq_len"]
        )

        return collator

    def create_callbacks(self) -> list[custom_callbacks.CustomCallback]:
        used_callbacks: list[custom_callbacks.CustomCallback] = []

        compute_metrics: bool = False
        for value in self._config["validation"]["metrics"].values():
            if value:
                compute_metrics = value
                break

        if compute_metrics:
            device: torch.device = torch.device(
                self._config["training"]["device"]
                if torch.cuda.is_available()
                else "cpu"
            )

            used_callbacks.append(
                custom_callbacks.MetricComputerCallback(
                    metrics_storage_dir=self._storage_path.joinpath("metrics"),
                    use_metrics=self._config["validation"]["metrics"],
                    device=device,
                    period=self._config["validation"]["period"],
                    threshold=self._config["validation"]["threshold"],
                    save_plot=self._config["validation"]["save_graphs"],
                )
            )

        used_callbacks.append(custom_callbacks.SaveLossHistoryCallback(self._storage_path.joinpath("loss"), True))

        used_callbacks.append(custom_callbacks.CheckpointCallback(self._storage_path.joinpath("checkpoint")))
        used_callbacks.append(custom_callbacks.SaveBestModelCallback(self._storage_path.joinpath("best_model")))

        return used_callbacks

    def create_training_args(self) -> TrainingArgs:
        return TrainingArgs(
            mode=self._config["model"]["head"]["mode"],
            n_epochs=self._config["training"]["n_epochs"],
            n_warmup_epochs=self._config["training"]["n_warmup_epochs"],
            device=torch.device(self._config["training"]["device"]),
            criterion=torch.nn.BCEWithLogitsLoss,
            batch_size=self._config["hyperparameters"]["batch_size"],
            **self._config["training"]["optimizer_parameters"],
        )

    def create_metric_computer(self) -> head_metrics.MultiLabelHeadMetricComputer:
        dataset: MultiLabelHeadDataset = self.create_dataset("validation")

        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=self._config["hyperparameters"]["batch_size"],
            collate_fn=self.create_collator(),
        )

        metric_computer: head_metrics.MultiLabelHeadMetricComputer = \
            head_metrics.MultiLabelHeadMetricComputer(
                self._config["validation"]["threshold"],
                dataloader,
                torch.device("cpu"),
                self._config["validation"]["metrics"],
                self._config["validation"]["confidence_interval"]["use"],
                self._config["validation"]["confidence_interval"]["interval_type"],
                self._config["validation"]["confidence_interval"]["confidence_level"],
            )

        return metric_computer


@head
class EncoderDecoderHeadFactory(HuggingFaceHeadFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def initialize(self) -> None:
        self._create_storage()

    def load_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self._config["tokenizer"]["from_huggingface"]:
            return AutoTokenizer.from_pretrained(
                self._config["tokenizer"]["path_to_saved_tokenizer"]
            )

        return AutoTokenizer.from_pretrained(
            os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
        )

    def create_model(self) -> transformers.EncoderDecoderModel:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        if self._config["model"]["from_encoder_decoder"]:
            from_config: bool = False
            if self._config["model"]["encoder"]["from_huggingface"]:
                path_to_saved_encoder: str = self._config["model"]["encoder"]["path_to_saved_model"]
            else:
                path_to_saved_encoder = os.path.abspath(self._config["model"]["encoder"]["path_to_saved_model"])

            if self._config["model"]["decoder"]["from_pretrained"]:
                if self._config["model"]["decoder"]["from_huggingface"]:
                    path_to_saved_decoder: str | pathlib.Path = self._config["model"]["decoder"]["path_to_saved_model"]
                else:
                    path_to_saved_decoder = os.path.abspath(self._config["model"]["decoder"]["path_to_saved_model"])
            else:
                decoder_config: GPT2Config = GPT2Config(
                    vocab_size=len(tokenizer.get_vocab()),
                    n_positions=self._config["hyperparameters"]["seq_len"],
                    n_embd=self._config["model"]["decoder"]["config"]["embedding_size"],
                    n_head=self._config["model"]["decoder"]["config"]["n_heads"],
                    n_layer=self._config["model"]["decoder"]["config"]["n_layers"],
                )
                from_config = True

                decoder: GPT2LMHeadModel = GPT2LMHeadModel(config=decoder_config)

                temp_dir: pathlib.Path = pathlib.Path(utils.get_cwd()).joinpath("_temp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                decoder.save_pretrained(temp_dir)

                path_to_saved_decoder = temp_dir

            self._model: transformers.EncoderDecoderModel = \
                transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
                    path_to_saved_encoder,
                    path_to_saved_decoder,
                )

            if from_config:
                shutil.rmtree(temp_dir)
        else:
            if self._config["model"]["saved_model"]["from_huggingface"]:
                path_to_saved_model: str = self._config["model"]["saved_model"]["path_to_saved_model"]
            else:
                path_to_saved_model = os.path.abspath(self._config["model"]["saved_model"]["path_to_saved_model"])

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
        max_length: int = self._config["hyperparameters"]["seq_len"]

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
            path=os.path.abspath(self._config[mode]["path_to_data"]),
            tokenizer=tokenizer,
            max_length=self._config["hyperparameters"]["seq_len"],
        )

        return dataset

    def create_training_args(self) -> transformers.Seq2SeqTrainingArguments:
        checkpoint_path: pathlib.Path = self._storage_path.joinpath("checkpoint")

        utils.create_dir(checkpoint_path)
        utils.delete_files(checkpoint_path)

        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.load_tokenizer()

        generation_config: GenerationConfig = GenerationConfig(
            max_new_tokens = self._config["hyperparameters"]["seq_len"],
            pad_token_id = tokenizer.pad_token_id,
            # do_sample = True,
            decoder_start_token_id = tokenizer.cls_token_id,
            # num_beams = 2,
            # top_k = 5,
        )

        args = transformers.Seq2SeqTrainingArguments(
            output_dir=str(checkpoint_path),
            overwrite_output_dir=True,
            num_train_epochs=self._config["training"]["n_epochs"],
            per_device_train_batch_size=self._config["hyperparameters"]["batch_size"],
            per_device_eval_batch_size=self._config["hyperparameters"]["batch_size"],
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
            prediction_loss_only=False,
            generation_config = generation_config,
        )

        return args

    def create_callbacks(self) -> list[transformers.TrainerCallback]:
        used_callbacks = []

        used_callbacks.append(
            hf_callbacks.SaveLossHistoryCallback(
                loss_storage_dir=self._storage_path.joinpath("loss"),
                save_plot=self._config["validation"]["save_graphs"],
            )
        )

        metrics_to_use: dict[str, bool] = self._config["validation"]["metrics"]

        compute_metrics: bool = False
        for value in metrics_to_use.values():
            if value:
                compute_metrics = value
                break

        if compute_metrics:
            used_callbacks.append(
                hf_callbacks.EncoderDecoderMetricPlotsCallback(
                    metrics_storage_dir=self._storage_path.joinpath("metrics"),
                    use_metrics=metrics_to_use,
                    save_plot=self._config["validation"]["save_graphs"],
                )
            )

        return used_callbacks

    def create_metric_computer(self) -> head_metrics.EncoderDecoderHeadMetricComputer:
        metric_computer: head_metrics.EncoderDecoderHeadMetricComputer = \
            head_metrics.EncoderDecoderHeadMetricComputer(
            tokenizer=self.load_tokenizer(),
            metrics_to_use=self._config["validation"]["metrics"],
            confidence_interval=self._config["validation"]["confidence_interval"]["use"],
            interval_type=self._config["validation"]["confidence_interval"]["interval_type"],
            confidence_level=self._config["validation"]["confidence_interval"]["confidence_level"],
        )

        return metric_computer
