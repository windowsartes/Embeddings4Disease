import os
import pathlib
import shutil
import typing as tp
from abc import ABC, abstractmethod
from math import ceil

import torch
import transformers
from transformers import (
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
try:
    import wandb
except ImportError:
    pass

from embeddings4disease.callbacks import callbacks
from embeddings4disease.data import collators, datasets
from embeddings4disease.metrics import metrics
from embeddings4disease.utils import utils


working_dir = pathlib.Path(os.getcwd())


class ArchitectureFactory(ABC):
    """
    Base class for model creation. It's usually automatically created by the AbstractFactory.

    Args:
        ABC (config (dict[str, tp.Any]): parsed config with all the required information.
    """

    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

        if config["wandb"]["use"]:
            wandb.login(key=config["wandb"]["api_key"])
            wandb.init(project=config["wandb"]["project"])

    @abstractmethod
    def create_model(self) -> PreTrainedModel:
        """
        This method can be used to create a proper model.

        Returns:
            PreTrainedModel: created model.
        """
        pass

    @abstractmethod
    def create_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        This method can be used to create a proper tokenizer.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: created tokenizer.
        """
        pass

    def create_storage(self) -> None:
        """
        This method is used to initialize storage dir in the case you need to store logs/graphs/etc somewhere.
        """
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])
        utils.create_dir(storage_path)

        self.storage_path: pathlib.Path = storage_path

    def create_collator(self) -> DataCollatorForLanguageModeling:
        """
        This method can be used to create a data collator which later will be using for training.

        Returns:
            DataCollatorForLanguageModeling: created data collator.
        """
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.create_tokenizer()
        )

        data_collator: DataCollatorForLanguageModeling = (
            DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15,
                return_tensors="pt",
            )
        )

        return data_collator

    def create_dataset(self, mode: str) -> LineByLineTextDataset:
        """
        This method can be used to create a training or validation dataset.
        All you need is to specify proper 'mode' value so factory can find proper information in the config.

        Args:
            mode (str): mode can be either 'validation' or 'training'

        Returns:
            LineByLineTextDataset: line by line dataset.
        """
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.create_tokenizer()
        )

        dataset: LineByLineTextDataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=working_dir.joinpath(self.config[mode]["path_to_data"]),
            block_size=self.config["hyperparameters"]["seq_len"] + 2,
        )

        return dataset

    def create_callbacks(self) -> list[TrainerCallback]:
        """
        This method can be used to create a list of callback based on the config.

        Returns:
            list[TrainerCallback]: list of selected callbacks.
        """
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.create_tokenizer()
        )

        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        used_callbacks: list[transformers.TrainerCallback] = []

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
                callbacks.MetricComputerCallback(
                    path_to_data=working_dir.joinpath(
                        self.config["validation"]["path_to_data"]
                    ),
                    metrics_storage_dir=storage_path.joinpath("metrics"),
                    tokenizer=tokenizer,
                    use_metrics=self.config["validation"]["metrics"],
                    device=device,
                    period=self.config["validation"]["period"],
                    top_k=self.config["validation"]["top_k"],
                    batch_size=self.config["hyperparameters"]["batch_size"],
                    seq_len=self.config["hyperparameters"]["seq_len"],
                    use_wandb=self.config["wandb"]["use"],
                ),
            )

        if self.config["validation"]["save_graphs"]:
            used_callbacks.append(
                callbacks.SaveGraphsCallback(
                    graph_storage_dir=storage_path.joinpath("graphs"),
                    metrics_storage_dir=storage_path.joinpath("metrics"),
                )
            )

        used_callbacks.append(
            callbacks.SaveLossHistoryCallback(
                loss_storage_dir=storage_path.joinpath("loss"),
            )
        )

        return used_callbacks

    def create_training_args(self) -> TrainingArguments:
        """
        This method can be used to create a training args based on the config.

        Returns:
            TrainingArguments: training arg which will be later used by trainer.
        """
        storage_path: pathlib.Path = working_dir.joinpath(self.config["storage_path"])

        checkpoint_path: pathlib.Path = storage_path.joinpath("checkpoint")

        utils.create_dir(checkpoint_path)
        utils.delete_files(checkpoint_path)

        training_args = TrainingArguments(
            output_dir=str(checkpoint_path),
            overwrite_output_dir=True,
            num_train_epochs=self.config["training"]["n_epochs"],
            per_device_train_batch_size=self.config["hyperparameters"]["batch_size"],
            per_device_eval_batch_size=self.config["hyperparameters"]["batch_size"],
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=self.config["training"]["n_checkpoints"],
            prediction_loss_only=True,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            report_to="wandb" if self.config["wandb"]["use"] else "none",
            **self.config["training"]["optimizer_parameters"],
        )

        return training_args

    def create_metric_computer(self) -> tuple[metrics.MetricComputer, dict[str, bool]]:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.create_tokenizer()
        )

        dataset: torch.utils.data.Dataset = datasets.CustomLineByLineDataset(
            working_dir.joinpath(self.config["validation"]["path_to_data"])
        )
        dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["hyperparameters"]["batch_size"],
            collate_fn=collators.MaskingCollator(
                tokenizer, self.config["hyperparameters"]["seq_len"]
            ),
        )

        metric_computer: metrics.MetricComputer = metrics.MetricComputer(
            tokenizer, self.config["validation"]["top_k"], dataloader
        )

        return (metric_computer, self.config["validation"]["metrics"])

    def set_warmup_epochs(
        self, training_args: TrainingArguments, dataset_train: LineByLineTextDataset
    ) -> None:
        """
        Training args don't have warmup_epochs argument, only warmup_steps, but number of steps depends on the
        length of dataloader so we need a way to compute warmup_steps by warmup_epochs.

        Args:
            training_args (TrainingArguments): Training args create by corresponding method.
            dataset_train (LineByLineTextDataset): Train dataset so we can get it's len to compute warmup_steps.
        """
        batch_size: int = self.config["hyperparameters"]["batch_size"]
        n_warmup_epochs: int = self.config["training"]["n_warmup_epochs"]

        training_args.warmup_steps = (
            ceil(len(dataset_train) / batch_size) * n_warmup_epochs
        )

    def optionally_save(self, trainer: Trainer) -> None:
        """
        Saves model in the case this option was selected in the config file.

        Args:
            trainer (Trainer): Trainer that trained your model.
        """
        if self.config["save_trained_model"]:
            trainer.save_model(self.storage_path.joinpath("saved_model"))


# C = tp.TypeVar("C", bound=ArchitectureFactory)

CLASS_REGISTER: dict[str, tp.Type[ArchitectureFactory]] = {}


def architecture(cls: tp.Type[ArchitectureFactory]) -> tp.Type[ArchitectureFactory]:
    """
    This decorator is used to register an architucture so Abstract Factory can create a proper model
    without any ifs inside its body.

    Args:
        cls (tp.Type[ArchitectureFactory]): a class to register.

    Returns:
        tp.Type[ArchitectureFactory]: registered class.
    """
    CLASS_REGISTER[cls.__name__[:-7]] = cls
    return cls


@architecture
class BERTFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.BertTokenizer:
        if self.config["tokenizer"]["use_pretrained"]:
            if self.config["tokenizer"]["path_to_saved_tokenizer"] is None:
                tokenizer: transformers.BertTokenizer = (
                    transformers.BertTokenizer.from_pretrained(
                        "windowsartes/bert_tokenizer"
                    )
                )
            else:
                tokenizer = transformers.BertTokenizer.from_pretrained(
                    working_dir.joinpath(
                        self.config["tokenizer"]["path_to_saved_tokenizer"]
                    )
                )
        else:
            tokenizer = transformers.BertTokenizer(
                vocab_file=working_dir.joinpath(
                    self.config["tokenizer"]["path_to_vocab_file"]
                ),
                do_lower_case=False,
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]",
            )

        return tokenizer

    def create_model(self) -> transformers.BertForMaskedLM:
        tokenizer: transformers.BertTokenizer = self.create_tokenizer()

        if self.config["model"]["use_pretrained"]:
            if self.config["model"]["path_to_saved_weights"] is None:
                print("load from huggingface")
                model: transformers.BertForMaskedLM = (
                    transformers.BertForMaskedLM.from_pretrained("windowsartes/bert")
                )
            else:
                print("load locally saved model")
                model = transformers.BertForMaskedLM.from_pretrained(
                    working_dir.joinpath(self.config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self.config["hyperparameters"]["seq_len"]
            config: transformers.BertConfig = transformers.BertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model = transformers.BertForMaskedLM(config=config)

        return model


@architecture
class ConvBERTFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.ConvBertTokenizer:
        if self.config["tokenizer"]["use_pretrained"]:
            if self.config["tokenizer"]["path_to_saved_tokenizer"] is None:
                tokenizer: transformers.ConvBertTokenizer = (
                    transformers.ConvBertTokenizer.from_pretrained(
                        "windowsartes/convbert_tokenizer"
                    )
                )
            else:
                tokenizer = transformers.ConvBertTokenizer.from_pretrained(
                    working_dir.joinpath(
                        self.config["tokenizer"]["path_to_saved_tokenizer"]
                    )
                )
        else:
            tokenizer = transformers.ConvBertTokenizer(
                vocab_file=working_dir.joinpath(
                    self.config["tokenizer"]["path_to_vocab_file"]
                ),
                do_lower_case=False,
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]",
            )

        return tokenizer

    def create_model(self) -> transformers.ConvBertForMaskedLM:
        tokenizer: transformers.ConvBertTokenizer = self.create_tokenizer()

        if self.config["model"]["use_pretrained"]:
            if self.config["model"]["path_to_saved_weights"] is None:
                model: transformers.ConvBertForMaskedLM = (
                    transformers.ConvBertForMaskedLM.from_pretrained(
                        "windowsartes/convbert"
                    )
                )
            else:
                model = transformers.ConvBertForMaskedLM.from_pretrained(
                    working_dir.joinpath(self.config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self.config["hyperparameters"]["seq_len"]
            config: transformers.ConvBertConfig = transformers.ConvBertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model = transformers.ConvBertForMaskedLM(config=config)

        return model


@architecture
class DeBERTaFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.DebertaTokenizerFast:
        if self.config["tokenizer"]["use_pretrained"]:
            if self.config["tokenizer"]["path_to_saved_tokenizer"] is None:
                tokenizer: transformers.DebertaTokenizerFast = (
                    transformers.DebertaTokenizerFast.from_pretrained(
                        "windowsartes/deberta_tokenizer_fast"
                    )
                )
            else:
                tokenizer = transformers.DebertaTokenizerFast.from_pretrained(
                    working_dir.joinpath(
                        self.config["tokenizer"]["path_to_saved_tokenizer"]
                    )
                )
        else:
            bert_tokenizer: transformers.BertTokenizerFast = (
                transformers.BertTokenizerFast(
                    vocab_file=working_dir.joinpath(
                        self.config["tokenizer"]["path_to_vocab_file"]
                    ),
                    do_lower_case=False,
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                )
            )
            bert_tokenizer.save_pretrained(working_dir.joinpath("_temp"))

            tokenizer = transformers.DebertaTokenizerFast.from_pretrained(
                working_dir.joinpath("_temp")
            )
            shutil.rmtree(working_dir.joinpath("_temp"))

        return tokenizer

    def create_model(self) -> transformers.DebertaForMaskedLM:
        tokenizer: transformers.DebertaTokenizerFast = self.create_tokenizer()

        if self.config["model"]["use_pretrained"]:
            if self.config["model"]["path_to_saved_weights"] is None:
                model: transformers.DebertaForMaskedLM = (
                    transformers.DebertaForMaskedLM.from_pretrained(
                        "windowsartes/deberta"
                    )
                )
            else:
                model = transformers.DebertaForMaskedLM.from_pretrained(
                    working_dir.joinpath(self.config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self.config["hyperparameters"]["seq_len"]
            config: transformers.DebertaConfig = transformers.DebertaConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model = transformers.DebertaForMaskedLM(config=config)

        return model


@architecture
class FNetFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.FNetTokenizerFast:
        if self.config["tokenizer"]["use_pretrained"]:
            if self.config["tokenizer"]["path_to_saved_tokenizer"] is None:
                tokenizer: transformers.FNetTokenizerFast = (
                    transformers.FNetTokenizerFast.from_pretrained(
                        "windowsartes/fnet_tokenizer_fast"
                    )
                )
            else:
                tokenizer = transformers.FNetTokenizerFast.from_pretrained(
                    working_dir.joinpath(
                        self.config["tokenizer"]["path_to_saved_tokenizer"]
                    )
                )
        else:
            bert_tokenizer: transformers.BertTokenizerFast = (
                transformers.BertTokenizerFast(
                    vocab_file=working_dir.joinpath(
                        self.config["tokenizer"]["path_to_vocab_file"]
                    ),
                    do_lower_case=False,
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                )
            )
            bert_tokenizer.save_pretrained(working_dir.joinpath("_temp"))

            tokenizer = transformers.FNetTokenizerFast.from_pretrained(
                working_dir.joinpath("_temp")
            )
            shutil.rmtree(working_dir.joinpath("_temp"))

        return tokenizer

    def create_model(self) -> transformers.FNetForMaskedLM:
        tokenizer: transformers.FNetTokenizerFast = self.create_tokenizer()

        if self.config["model"]["use_pretrained"]:
            if self.config["model"]["path_to_saved_weights"] is None:
                model: transformers.FNetForMaskedLM = (
                    transformers.FNetForMaskedLM.from_pretrained("windowsartes/fnet")
                )
            else:
                model = transformers.FNetForMaskedLM.from_pretrained(
                    working_dir.joinpath(self.config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self.config["hyperparameters"]["seq_len"]
            config: transformers.FNetConfig = transformers.FNetConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model = transformers.FNetForMaskedLM(config=config)

        return model


@architecture
class FunnelTransformerFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.FunnelTokenizer:
        if self.config["tokenizer"]["use_pretrained"]:
            if self.config["tokenizer"]["path_to_saved_tokenizer"] is None:
                tokenizer: transformers.FunnelTokenizer = (
                    transformers.FunnelTokenizer.from_pretrained(
                        "windowsartes/funnel_tokenizer"
                    )
                )
            else:
                tokenizer = transformers.FunnelTokenizer.from_pretrained(
                    working_dir.joinpath(
                        self.config["tokenizer"]["path_to_saved_tokenizer"]
                    )
                )
        else:
            tokenizer = transformers.FunnelTokenizer(
                vocab_file=working_dir.joinpath(
                    self.config["tokenizer"]["path_to_vocab_file"]
                ),
                do_lower_case=False,
                do_basic_tokenize=True,
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]",
            )

        return tokenizer

    def create_model(self) -> transformers.FunnelForMaskedLM:
        tokenizer: transformers.FunnelTokenizer = self.create_tokenizer()

        if self.config["model"]["use_pretrained"]:
            if self.config["model"]["path_to_saved_weights"] is None:
                model: transformers.FunnelForMaskedLM = (
                    transformers.FunnelForMaskedLM.from_pretrained(
                        "windowsartes/funnel"
                    )
                )
            else:
                model = transformers.FunnelForMaskedLM.from_pretrained(
                    working_dir.joinpath(self.config["model"]["path_to_saved_weights"])
                )
        else:
            config: transformers.FunnelConfig = transformers.FunnelConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                **self.config["model"]["config"],
            )
            model = transformers.FunnelForMaskedLM(config=config)

        return model


@architecture
class MobileBERTFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.MobileBertTokenizer:
        if self.config["tokenizer"]["use_pretrained"]:
            if self.config["tokenizer"]["path_to_saved_tokenizer"] is None:
                tokenizer: transformers.MobileBertTokenizer = (
                    transformers.MobileBertTokenizer.from_pretrained(
                        "windowsartes/mobilebert_tokenizer"
                    )
                )
            else:
                tokenizer = transformers.MobileBertTokenizer.from_pretrained(
                    working_dir.joinpath(
                        self.config["tokenizer"]["path_to_saved_tokenizer"]
                    )
                )
        else:
            tokenizer = transformers.MobileBertTokenizer(
                vocab_file=working_dir.joinpath(
                    self.config["tokenizer"]["path_to_vocab_file"]
                ),
                do_lower_case=False,
                do_basic_tokenize=True,
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]",
            )

        return tokenizer

    def create_model(self) -> transformers.MobileBertForMaskedLM:
        tokenizer: transformers.FunnelTokenizer = self.create_tokenizer()

        if self.config["model"]["use_pretrained"]:
            if self.config["model"]["path_to_saved_weights"] is None:
                model: transformers.MobileBertForMaskedLM = (
                    transformers.MobileBertForMaskedLM.from_pretrained(
                        "windowsartes/mobilebert"
                    )
                )
            else:
                model = transformers.MobileBertForMaskedLM.from_pretrained(
                    working_dir.joinpath(self.config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self.config["hyperparameters"]["seq_len"]
            config: transformers.MobileBertConfig = transformers.MobileBertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self.config["model"]["config"],
            )
            model = transformers.MobileBertForMaskedLM(config=config)

        return model


@architecture
class RoBERTaFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.RobertaTokenizerFast:
        if self.config["tokenizer"]["use_pretrained"]:
            if self.config["tokenizer"]["path_to_saved_tokenizer"] is None:
                tokenizer: transformers.RobertaTokenizerFast = (
                    transformers.RobertaTokenizerFast.from_pretrained(
                        "windowsartes/roberta_tokenizer_fast"
                    )
                )
            else:
                tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
                    working_dir.joinpath(
                        self.config["tokenizer"]["path_to_saved_tokenizer"]
                    )
                )
        else:
            bert_tokenizer: transformers.BertTokenizerFast = (
                transformers.BertTokenizerFast(
                    vocab_file=working_dir.joinpath(
                        self.config["tokenizer"]["path_to_vocab_file"]
                    ),
                    do_lower_case=False,
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                )
            )
            bert_tokenizer.save_pretrained(working_dir.joinpath("_temp"))

            tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
                working_dir.joinpath("_temp")
            )
            shutil.rmtree(working_dir.joinpath("_temp"))

        return tokenizer

    def create_model(self) -> transformers.RobertaForMaskedLM:
        tokenizer: transformers.RobertaTokenizerFast = self.create_tokenizer()

        if self.config["model"]["use_pretrained"]:
            if self.config["model"]["path_to_saved_weights"] is None:
                model: transformers.RobertaForMaskedLM = (
                    transformers.RobertaForMaskedLM.from_pretrained(
                        "windowsartes/roberta"
                    )
                )
            else:
                model = transformers.RobertaForMaskedLM.from_pretrained(
                    working_dir.joinpath(self.config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self.config["hyperparameters"]["seq_len"]
            config: transformers.RobertaConfig = transformers.RobertaConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 4,
                **self.config["model"]["config"],
            )
            model = transformers.RobertaForMaskedLM(config=config)

        return model


@architecture
class RoFormerFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.RoFormerTokenizer:
        if self.config["tokenizer"]["use_pretrained"]:
            if self.config["tokenizer"]["path_to_saved_tokenizer"] is None:
                tokenizer: transformers.RoFormerTokenizer = (
                    transformers.RoFormerTokenizer.from_pretrained(
                        "windowsartes/roformer_tokenizer"
                    )
                )
            else:
                tokenizer = transformers.RoFormerTokenizer.from_pretrained(
                    working_dir.joinpath(
                        self.config["tokenizer"]["path_to_saved_tokenizer"]
                    )
                )
        else:
            tokenizer = transformers.RoFormerTokenizer(
                vocab_file=working_dir.joinpath(
                    self.config["tokenizer"]["path_to_vocab_file"]
                ),
                do_lower_case=False,
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]",
            )

        return tokenizer

    def create_model(self) -> transformers.RoFormerForMaskedLM:
        tokenizer: transformers.RoFormerTokenizer = self.create_tokenizer()

        if self.config["model"]["use_pretrained"]:
            if self.config["model"]["path_to_saved_weights"] is None:
                model: transformers.RoFormerForMaskedLM = (
                    transformers.RoFormerForMaskedLM.from_pretrained(
                        "windowsartes/roformer"
                    )
                )
            else:
                model = transformers.RoFormerForMaskedLM.from_pretrained(
                    working_dir.joinpath(self.config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self.config["hyperparameters"]["seq_len"]
            config: transformers.RoFormerConfig = transformers.RoFormerConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 4,
                **self.config["model"]["config"],
            )
            model = transformers.RoFormerForMaskedLM(config=config)

        return model


@architecture
class XLMRoBERTaFactory(ArchitectureFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.XLMRobertaTokenizerFast:
        if self.config["tokenizer"]["use_pretrained"]:
            if self.config["tokenizer"]["path_to_saved_tokenizer"] is None:
                tokenizer: transformers.XLMRobertaTokenizerFast = (
                    transformers.XLMRobertaTokenizerFast.from_pretrained(
                        "windowsartes/xlmroberta_tokenizer_fast"
                    )
                )
            else:
                tokenizer = transformers.XLMRobertaTokenizerFast.from_pretrained(
                    working_dir.joinpath(
                        self.config["tokenizer"]["path_to_saved_tokenizer"]
                    )
                )
        else:
            bert_tokenizer: transformers.BertTokenizerFast = (
                transformers.BertTokenizerFast(
                    vocab_file=working_dir.joinpath(
                        self.config["tokenizer"]["path_to_vocab_file"]
                    ),
                    do_lower_case=False,
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                )
            )
            bert_tokenizer.save_pretrained(working_dir.joinpath("_temp"))

            tokenizer = transformers.XLMRobertaTokenizerFast.from_pretrained(
                working_dir.joinpath("_temp")
            )
            shutil.rmtree(working_dir.joinpath("_temp"))

        return tokenizer

    def create_model(self) -> transformers.XLMRobertaForMaskedLM:
        tokenizer: transformers.XLMRobertaTokenizerFast = self.create_tokenizer()

        if self.config["model"]["use_pretrained"]:
            if self.config["model"]["path_to_saved_weights"] is None:
                model: transformers.XLMRobertaForMaskedLM = (
                    transformers.XLMRobertaForMaskedLM.from_pretrained(
                        "windowsartes/xlmroberta"
                    )
                )
            else:
                model = transformers.XLMRobertaForMaskedLM.from_pretrained(
                    working_dir.joinpath(self.config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self.config["hyperparameters"]["seq_len"]
            config: transformers.XLMRobertaConfig = transformers.XLMRobertaConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 4,
                **self.config["model"]["config"],
            )
            model = transformers.XLMRobertaForMaskedLM(config=config)

        return model
