import os
import pathlib
import shutil
import typing as tp
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
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
    import wandb  # type: ignore
except ImportError:
    warnings.warn(ImportWarning("wandb isn't installed so it won't be used."))
    wandb_installed: bool = False
else:
    wandb_installed = True

from embeddings4disease.callbacks import hf_callbacks
from embeddings4disease.data import collators, datasets
from embeddings4disease.metrics import backbone_metrics
from embeddings4disease.utils import utils


class BackboneFactory(ABC):
    """
    Base class for model creation. It's usually automatically created by the AbstractFactory.

    Args:
        ABC (config (dict[str, tp.Any]): parsed config with all the required information.
    """

    def __init__(self, config: dict[str, tp.Any]):
        self._config: dict[str, tp.Any] = config

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

    def initialize(self) -> None:
        """
        This method is used if we need to initialize some features in the case of training.
        """
        self._create_storage()
        self._log_into_wandb()

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

    def _log_into_wandb(self) -> None:
        """
        This method is used to log into wandb.
        """
        if wandb_installed and self._config["wandb"]["use"]:
            wandb.login(key=self._config["wandb"]["api_key"])
            wandb.init(project=self._config["wandb"]["project"])

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
            file_path=os.path.abspath(self._config[mode]["path_to_data"]),
            block_size=self._config["hyperparameters"]["seq_len"] + 2,
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
        used_callbacks: list[transformers.TrainerCallback] = []

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
                hf_callbacks.MetricComputerCallback(
                    path_to_data=os.path.abspath(
                        self._config["validation"]["path_to_data"]
                    ),
                    metrics_storage_dir=self._storage_path.joinpath("metrics"),
                    tokenizer=tokenizer,
                    use_metrics=self._config["validation"]["metrics"],
                    device=device,
                    period=self._config["validation"]["period"],
                    top_k=self._config["validation"]["top_k"],
                    batch_size=self._config["hyperparameters"]["batch_size"],
                    seq_len=self._config["hyperparameters"]["seq_len"],
                    use_wandb=wandb_installed and self._config["wandb"]["use"],
                    save_plot=self._config["validation"]["save_graphs"],
                ),
            )

        used_callbacks.append(
            hf_callbacks.SaveLossHistoryCallback(
                loss_storage_dir=self._storage_path.joinpath("loss"),
                save_plot=self._config["validation"]["save_graphs"],
            )
        )

        return used_callbacks

    def create_training_args(self) -> TrainingArguments:
        """
        This method can be used to create a training args based on the config.

        Returns:
            TrainingArguments: training arg which will be later used by trainer.
        """
        checkpoint_path: pathlib.Path = self._storage_path.joinpath("checkpoint")

        utils.create_dir(checkpoint_path)
        utils.delete_files(checkpoint_path)

        training_args = TrainingArguments(
            output_dir=str(checkpoint_path),
            overwrite_output_dir=True,
            num_train_epochs=self._config["training"]["n_epochs"],
            per_device_train_batch_size=self._config["hyperparameters"]["batch_size"],
            per_device_eval_batch_size=self._config["hyperparameters"]["batch_size"],
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=self._config["training"]["n_checkpoints"],
            prediction_loss_only=True,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            report_to=(
                "wandb" if (wandb_installed and self._config["wandb"]["use"]) else "none"
            ),
            **self._config["training"]["optimizer_parameters"],
        )

        return training_args

    def create_metric_computer(self) -> backbone_metrics.MLMMetricComputer:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.create_tokenizer()
        )

        dataset: torch.utils.data.Dataset = datasets.CustomLineByLineDataset(
            os.path.abspath(self._config["validation"]["path_to_data"])
        )
        dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._config["hyperparameters"]["batch_size"],
            collate_fn=collators.MaskingCollator(
                tokenizer, self._config["hyperparameters"]["seq_len"]
            ),
        )

        metric_computer: backbone_metrics.MLMMetricComputer = backbone_metrics.MLMMetricComputer(
            tokenizer,
            self._config["validation"]["top_k"],
            dataloader,
            self._config["validation"]["metrics"],
            self._config["validation"]["confidence_interval"]["use"],
            self._config["validation"]["confidence_interval"]["interval_type"],
            self._config["validation"]["confidence_interval"]["confidence_level"],
        )

        return metric_computer

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
        batch_size: int = self._config["hyperparameters"]["batch_size"]
        n_warmup_epochs: int = self._config["training"]["n_warmup_epochs"]

        training_args.warmup_steps = (
            ceil(len(dataset_train) / batch_size) * n_warmup_epochs
        )

    def optionally_save(self, trainer: Trainer) -> None:
        """
        Saves model in the case this option was selected in the config file.

        Args:
            trainer (Trainer): Trainer that trained your model.
        """
        if self._config["save_trained_model"]:
            trainer.save_model(self._storage_path.joinpath("saved_model"))


BACKBONE_REGISTER: dict[str, tp.Type[BackboneFactory]] = {}


def backbone(cls: tp.Type[BackboneFactory]) -> tp.Type[BackboneFactory]:
    """
    This decorator is used to register an architucture so Abstract Factory can create a proper model
    without any ifs inside its body.

    Args:
        cls (tp.Type[ArchitectureFactory]): a class to register.

    Returns:
        tp.Type[ArchitectureFactory]: registered class.
    """
    BACKBONE_REGISTER[cls.__name__[:-7]] = cls
    return cls


@backbone
class BERTFactory(BackboneFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.BertTokenizer:
        if self._config["tokenizer"]["use_pretrained"]:
            if self._config["tokenizer"]["from_huggingface"]:
                tokenizer: transformers.BertTokenizer = (
                    transformers.BertTokenizer.from_pretrained(
                        self._config["tokenizer"]["path_to_saved_tokenizer"],
                    )
                )
            else:
                tokenizer = transformers.BertTokenizer.from_pretrained(
                    os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
                )
        else:
            tokenizer = transformers.BertTokenizer(
                vocab_file=os.path.abspath(
                    self._config["tokenizer"]["path_to_vocab_file"]
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

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.BertForMaskedLM = (
                    transformers.BertForMaskedLM.from_pretrained("windowsartes/bert")
                )
            else:
                model = transformers.BertForMaskedLM.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self._config["hyperparameters"]["seq_len"]
            config: transformers.BertConfig = transformers.BertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self._config["model"]["config"],
            )
            model = transformers.BertForMaskedLM(config=config)

        return model


@backbone
class BERTWithNSPFactory(BERTFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_model(self) -> transformers.BertForPreTraining:
        tokenizer: transformers.BertTokenizer = self.create_tokenizer()

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.BertForPreTraining = (
                    transformers.BertForPreTraining.from_pretrained("windowsartes/bert_with_nsp")
                )
            else:
                model = transformers.BertForPreTraining.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self._config["hyperparameters"]["seq_len"]
            config: transformers.BertConfig = transformers.BertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=2*seq_len + 3,
                **self._config["model"]["config"],
            )
            model = transformers.BertForPreTraining(config=config)

        return model

    def create_dataset(self, mode: str) -> datasets.CustomTextDatasetForNextSentencePrediction:
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

        seq_len: int = self._config["hyperparameters"]["seq_len"]

        dataset: datasets.CustomTextDatasetForNextSentencePrediction = \
            datasets.CustomTextDatasetForNextSentencePrediction(
                tokenizer=tokenizer,
                seq_len=seq_len,
                file_path=os.path.abspath(self._config[mode]["path_to_data"]),
                block_size=2*seq_len + 3,
            )

        return dataset

@backbone
class ConvBERTFactory(BackboneFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.ConvBertTokenizer:
        if self._config["tokenizer"]["use_pretrained"]:
            if self._config["tokenizer"]["from_huggingface"]:
                tokenizer: transformers.BertTokenizer = (
                    transformers.ConvBertTokenizer.from_pretrained(
                        self._config["tokenizer"]["path_to_saved_tokenizer"],
                    )
                )
            else:
                tokenizer = transformers.ConvBertTokenizer.from_pretrained(
                    os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
                )
        else:
            tokenizer = transformers.ConvBertTokenizer(
                vocab_file=os.path.abspath(
                    self._config["tokenizer"]["path_to_vocab_file"]
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

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.ConvBertForMaskedLM = (
                    transformers.ConvBertForMaskedLM.from_pretrained(
                        "windowsartes/convbert"
                    )
                )
            else:
                model = transformers.ConvBertForMaskedLM.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self._config["hyperparameters"]["seq_len"]
            config: transformers.ConvBertConfig = transformers.ConvBertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self._config["model"]["config"],
            )
            model = transformers.ConvBertForMaskedLM(config=config)

        return model


@backbone
class DeBERTaFactory(BackboneFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

        self._explicitly_added_tokens: bool = False

    def create_tokenizer(self) -> transformers.DebertaTokenizerFast:
        if self._config["tokenizer"]["use_pretrained"]:
            if self._config["tokenizer"]["from_huggingface"]:
                tokenizer: transformers.BertTokenizer = (
                    transformers.DebertaTokenizerFast.from_pretrained(
                        self._config["tokenizer"]["path_to_saved_tokenizer"],
                    )
                )
            else:
                tokenizer = transformers.DebertaTokenizerFast.from_pretrained(
                    os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
                )
        else:
            # don't know why, but we have to explicitly add [UNK] token to the vocab file
            if not self._explicitly_added_tokens:
                with open(self._config["tokenizer"]["path_to_vocab_file"], "a") as _vocab_file:
                    _vocab_file.write("[UNK]\n")
                    _vocab_file.write("[SEP]\n")
                    _vocab_file.write("[PAD]\n")
                    _vocab_file.write("[CLS]\n")
                    _vocab_file.write("[MASK]\n")

                self._explicitly_added_tokens = True

            bert_tokenizer: transformers.BertTokenizerFast = (
                transformers.BertTokenizerFast(
                    vocab_file=os.path.abspath(
                        self._config["tokenizer"]["path_to_vocab_file"]
                    ),
                    do_lower_case=False,
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                )
            )
            temp_dir: pathlib.Path = pathlib.Path(utils.get_cwd()).joinpath("_temp")

            bert_tokenizer.save_pretrained(temp_dir)

            tokenizer = transformers.DebertaTokenizerFast.from_pretrained(temp_dir)
            shutil.rmtree(temp_dir)

        return tokenizer

    def create_model(self) -> transformers.DebertaForMaskedLM:
        tokenizer: transformers.DebertaTokenizerFast = self.create_tokenizer()

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.DebertaForMaskedLM = (
                    transformers.DebertaForMaskedLM.from_pretrained(
                        "windowsartes/deberta"
                    )
                )
            else:
                model = transformers.DebertaForMaskedLM.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self._config["hyperparameters"]["seq_len"]
            config: transformers.DebertaConfig = transformers.DebertaConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self._config["model"]["config"],
            )
            model = transformers.DebertaForMaskedLM(config=config)

        return model


@backbone
class FNetFactory(BackboneFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

        self._explicitly_added_tokens: bool = False

    def create_tokenizer(self) -> transformers.FNetTokenizerFast:
        if self._config["tokenizer"]["use_pretrained"]:
            if self._config["tokenizer"]["from_huggingface"]:
                tokenizer: transformers.BertTokenizer = (
                    transformers.FNetTokenizerFast.from_pretrained(
                        self._config["tokenizer"]["path_to_saved_tokenizer"],
                    )
                )
            else:
                tokenizer = transformers.FNetTokenizerFast.from_pretrained(
                    os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
                )
        else:
            # don't know why, but we have to explicitly add [UNK] token to the vocab file
            if not self._explicitly_added_tokens:
                with open(self._config["tokenizer"]["path_to_vocab_file"], "a") as _vocab_file:
                    _vocab_file.write("[UNK]\n")
                    _vocab_file.write("[SEP]\n")
                    _vocab_file.write("[PAD]\n")
                    _vocab_file.write("[CLS]\n")
                    _vocab_file.write("[MASK]\n")

                self._explicitly_added_tokens = True

            bert_tokenizer: transformers.BertTokenizerFast = (
                transformers.BertTokenizerFast(
                    vocab_file=os.path.abspath(
                        self._config["tokenizer"]["path_to_vocab_file"]
                    ),
                    do_lower_case=False,
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                )
            )
            temp_dir: pathlib.Path = pathlib.Path(utils.get_cwd()).joinpath("_temp")

            bert_tokenizer.save_pretrained(temp_dir)

            tokenizer = transformers.FNetTokenizerFast.from_pretrained(temp_dir)
            shutil.rmtree(temp_dir)

        return tokenizer

    def create_model(self) -> transformers.FNetForMaskedLM:
        tokenizer: transformers.FNetTokenizerFast = self.create_tokenizer()

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.FNetForMaskedLM = (
                    transformers.FNetForMaskedLM.from_pretrained("windowsartes/fnet")
                )
            else:
                model = transformers.FNetForMaskedLM.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self._config["hyperparameters"]["seq_len"]
            config: transformers.FNetConfig = transformers.FNetConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self._config["model"]["config"],
            )
            model = transformers.FNetForMaskedLM(config=config)

        return model


@backbone
class FunnelTransformerFactory(BackboneFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.FunnelTokenizer:
        if self._config["tokenizer"]["use_pretrained"]:
            if self._config["tokenizer"]["from_huggingface"]:
                tokenizer: transformers.FunnelTokenizer = (
                    transformers.FunnelTokenizer.from_pretrained(
                        self._config["tokenizer"]["path_to_saved_tokenizer"],
                    )
                )
            else:
                tokenizer = transformers.FunnelTokenizer.from_pretrained(
                    os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
                )
        else:
            tokenizer = transformers.FunnelTokenizer(
                vocab_file=os.path.abspath(
                    self._config["tokenizer"]["path_to_vocab_file"]
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

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.FunnelForMaskedLM = (
                    transformers.FunnelForMaskedLM.from_pretrained(
                        "windowsartes/funnel"
                    )
                )
            else:
                model = transformers.FunnelForMaskedLM.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            config: transformers.FunnelConfig = transformers.FunnelConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                **self._config["model"]["config"],
            )
            model = transformers.FunnelForMaskedLM(config=config)

        return model


@backbone
class MobileBERTFactory(BackboneFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.MobileBertTokenizer:
        if self._config["tokenizer"]["use_pretrained"]:
            if self._config["tokenizer"]["from_huggingface"]:
                tokenizer: transformers.MobileBertTokenizer = (
                    transformers.MobileBertTokenizer.from_pretrained(
                        self._config["tokenizer"]["path_to_saved_tokenizer"],
                    )
                )
            else:
                tokenizer = transformers.MobileBertTokenizer.from_pretrained(
                    os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
                )
        else:
            tokenizer = transformers.MobileBertTokenizer(
                vocab_file=os.path.abspath(
                    self._config["tokenizer"]["path_to_vocab_file"]
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

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.MobileBertForMaskedLM = (
                    transformers.MobileBertForMaskedLM.from_pretrained(
                        "windowsartes/mobilebert"
                    )
                )
            else:
                model = transformers.MobileBertForMaskedLM.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self._config["hyperparameters"]["seq_len"]
            config: transformers.MobileBertConfig = transformers.MobileBertConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 2,
                **self._config["model"]["config"],
            )
            model = transformers.MobileBertForMaskedLM(config=config)

        return model


@backbone
class RoBERTaFactory(BackboneFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

        self._explicitly_added_tokens: bool = False

    def create_tokenizer(self) -> transformers.RobertaTokenizerFast:
        if self._config["tokenizer"]["use_pretrained"]:
            if self._config["tokenizer"]["from_huggingface"]:
                tokenizer: transformers.BertTokenizerFast = (
                    transformers.BertTokenizerFast.from_pretrained(
                        self._config["tokenizer"]["path_to_saved_tokenizer"],
                    )
                )
            else:
                tokenizer = transformers.BertTokenizerFast.from_pretrained(
                    os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
                )
        else:
            # don't know why, but we have to explicitly add [UNK] token to the vocab file
            if not self._explicitly_added_tokens:
                with open(self._config["tokenizer"]["path_to_vocab_file"], "a") as _vocab_file:
                    _vocab_file.write("[UNK]\n")
                    _vocab_file.write("[SEP]\n")
                    _vocab_file.write("[PAD]\n")
                    _vocab_file.write("[CLS]\n")
                    _vocab_file.write("[MASK]\n")

                self._explicitly_added_tokens = True

            bert_tokenizer: transformers.BertTokenizerFast = (
                transformers.BertTokenizerFast(
                    vocab_file=os.path.abspath(
                        self._config["tokenizer"]["path_to_vocab_file"]
                    ),
                    do_lower_case=False,
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                )
            )
            temp_dir: pathlib.Path = pathlib.Path(utils.get_cwd()).joinpath("_temp")

            bert_tokenizer.save_pretrained(temp_dir)

            tokenizer = transformers.RobertaTokenizerFast.from_pretrained(temp_dir)
            shutil.rmtree(temp_dir)

        return tokenizer

    def create_model(self) -> transformers.RobertaForMaskedLM:
        tokenizer: transformers.RobertaTokenizerFast = self.create_tokenizer()

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.RobertaForMaskedLM = (
                    transformers.RobertaForMaskedLM.from_pretrained(
                        "windowsartes/roberta"
                    )
                )
            else:
                model = transformers.RobertaForMaskedLM.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self._config["hyperparameters"]["seq_len"]
            config: transformers.RobertaConfig = transformers.RobertaConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 4,
                **self._config["model"]["config"],
            )
            model = transformers.RobertaForMaskedLM(config=config)

        return model


@backbone
class RoFormerFactory(BackboneFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

    def create_tokenizer(self) -> transformers.RoFormerTokenizer:
        if self._config["tokenizer"]["use_pretrained"]:
            if self._config["tokenizer"]["from_huggingface"]:
                tokenizer: transformers.RoFormerTokenizer = (
                    transformers.RoFormerTokenizer.from_pretrained(
                        self._config["tokenizer"]["path_to_saved_tokenizer"],
                    )
                )
            else:
                tokenizer = transformers.RoFormerTokenizer.from_pretrained(
                    os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
                )
        else:
            tokenizer = transformers.RoFormerTokenizer(
                vocab_file=os.path.abspath(
                    self._config["tokenizer"]["path_to_vocab_file"]
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

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.RoFormerForMaskedLM = (
                    transformers.RoFormerForMaskedLM.from_pretrained(
                        "windowsartes/roformer"
                    )
                )
            else:
                model = transformers.RoFormerForMaskedLM.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self._config["hyperparameters"]["seq_len"]
            config: transformers.RoFormerConfig = transformers.RoFormerConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 4,
                **self._config["model"]["config"],
            )
            model = transformers.RoFormerForMaskedLM(config=config)

        return model


@backbone
class XLMRoBERTaFactory(BackboneFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

        self._explicitly_added_tokens: bool = False

    def create_tokenizer(self) -> transformers.XLMRobertaTokenizerFast:
        if self._config["tokenizer"]["use_pretrained"]:
            if self._config["tokenizer"]["from_huggingface"]:
                tokenizer: transformers.XLMRobertaTokenizerFast = (
                    transformers.XLMRobertaTokenizerFast.from_pretrained(
                        self._config["tokenizer"]["path_to_saved_tokenizer"],
                    )
                )
            else:
                tokenizer = transformers.XLMRobertaTokenizerFast.from_pretrained(
                    os.path.abspath(self._config["tokenizer"]["path_to_saved_tokenizer"])
                )
        else:
            if not self._explicitly_added_tokens:
                with open(self._config["tokenizer"]["path_to_vocab_file"], "a") as _vocab_file:
                    _vocab_file.write("[UNK]\n")
                    _vocab_file.write("[SEP]\n")
                    _vocab_file.write("[PAD]\n")
                    _vocab_file.write("[CLS]\n")
                    _vocab_file.write("[MASK]\n")

                self._explicitly_added_tokens = True

            bert_tokenizer: transformers.BertTokenizerFast = (
                transformers.BertTokenizerFast(
                    vocab_file=os.path.abspath(
                        self._config["tokenizer"]["path_to_vocab_file"]
                    ),
                    do_lower_case=False,
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                )
            )
            temp_dir: pathlib.Path = pathlib.Path(utils.get_cwd()).joinpath("_temp")

            bert_tokenizer.save_pretrained(temp_dir)

            tokenizer = transformers.XLMRobertaTokenizerFast.from_pretrained(temp_dir)
            shutil.rmtree(temp_dir)

        return tokenizer

    def create_model(self) -> transformers.XLMRobertaForMaskedLM:
        tokenizer: transformers.XLMRobertaTokenizerFast = self.create_tokenizer()

        if self._config["model"]["use_pretrained"]:
            if self._config["model"]["path_to_saved_weights"] is None:
                model: transformers.XLMRobertaForMaskedLM = (
                    transformers.XLMRobertaForMaskedLM.from_pretrained(
                        "windowsartes/xlmroberta"
                    )
                )
            else:
                model = transformers.XLMRobertaForMaskedLM.from_pretrained(
                    os.path.abspath(self._config["model"]["path_to_saved_weights"])
                )
        else:
            seq_len: int = self._config["hyperparameters"]["seq_len"]
            config: transformers.XLMRobertaConfig = transformers.XLMRobertaConfig(
                vocab_size=tokenizer.vocab_size + len(tokenizer.all_special_tokens),
                max_position_embeddings=seq_len + 4,
                **self._config["model"]["config"],
            )
            model = transformers.XLMRobertaForMaskedLM(config=config)

        return model
