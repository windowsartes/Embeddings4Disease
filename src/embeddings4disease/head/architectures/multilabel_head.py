import warnings
from collections import OrderedDict

import torch
import transformers
from torch import nn
from transformers import PreTrainedModel


class LinearBlock(nn.Module):
    """
    Simple linear block with Linear layer, optional batch normaliaztion, rele activation and and optional drouout layer.
    Commonly used by the MultiLabelHead.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_dropout: bool = True,
        dropout_rate: float = 0.1,
        use_normalization: bool = True,
    ):
        super().__init__()

        self.__block = nn.Sequential(
            OrderedDict([
                ("linear", nn.Linear(input_size, output_size, bias=False if use_normalization else True)),
                ("normalization", nn.BatchNorm1d(output_size) if use_normalization else nn.Identity()),
                ("activation", nn.ReLU()),
                ("dropout", nn.Dropout1d(dropout_rate) if use_dropout else nn.Identity()),
            ])
        )

    @property
    def block(self) -> nn.Sequential:
        return self.__block

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.__block(input_tensor)  # type: ignore


class MultiLabelHead(nn.Module):
    """
    Model we want to use with pretrained backbone to predict new diseases person will get in the future.

    Args:
        backbone (PreTrainedModel): pretrained BERT-like model you want to use as a feature extractor.
        n_classes (int): maximum number of classes.
        n_last_layers (int, optional): number of last [CLS] embeddings you want to use for the prediction. Defaults to 4.
        aggregation (str, optional): [CLS] embeddings aggregation. Can be either 'concatenation' or 'addition'.
            Defaults to 'concatenation'.
        mode (str, optional): 'transfer learning' if you want to freeze the backbone or 'fine tuning' if you want to train it.
            Defaults to 'transfer learning'.
        hidden_sizes (tuple[int, ...] | list[int], optional): Linear blocks hidden sizes. Defaults to (1024, 512, 256,).
        hidden_use_dropout (bool, optional): Use dropout inside linear blocks or not. Defaults to True.
        hidden_dropout_rate (float, optional): dropout's rate. Will be ignored if hidden_use_dropout is False. Defaults to 0.1.
        hidden_use_normalization (bool, optional): Use normalization inside linear blocks or not. Defaults to True.
    """
    def __init__(
        self,
        backbone: PreTrainedModel,
        n_classes: int,
        n_last_layers: int = 4,
        aggregation: str = "concatenation",
        mode: str = "transfer learning",
        hidden_sizes: tuple[int, ...] | list[int] = (1024, 512, 256,),
        hidden_use_dropout: bool = True,
        hidden_dropout_rate: float = 0.1,
        hidden_use_normalization: bool = True,
    ):
        super().__init__()

        self.__backbone: PreTrainedModel = backbone
        self.__n_classes: int = n_classes
        self.__n_last_layers: int = min(n_last_layers, len(self.__backbone.bert.encoder.layer))

        if mode not in ["transfer learning", "fine tuning"]:
            warnings.warn(
                f"{mode} doen't supported; 'transfer learning' mode will be used",
                UserWarning,
            )

            self.__mode: str = "transfer learning"
        else:
            self.__mode = mode

        if self.__mode == "transfer learning":
            for param in self.__backbone.parameters():
                param.requires_grad = False
            self.__backbone.eval()
        elif self.__mode == "fine tuning":
            self.__backbone.train()

        if aggregation not in ["concatenation", "addition"]:
            warnings.warn(
                f"{aggregation} doen't supported; 'concatenation' aggregation will be used",
                UserWarning,
            )

            self.__aggregation: str = "concatenation"
        else:
            self.__aggregation = aggregation

        embedding_size: int = self.__backbone.bert.embeddings.word_embeddings.embedding_dim

        if self.__aggregation == "concatenation":
            input_size: int = embedding_size * self.__n_last_layers
        elif self.__aggregation == "addition":
            input_size = embedding_size

        self.__head = nn.Sequential(
            OrderedDict([
                (
                    "input_block",
                    LinearBlock(
                        input_size,
                        hidden_sizes[0],
                        hidden_use_dropout,
                        hidden_dropout_rate,
                        hidden_use_normalization,
                    )
                ),
                (
                    "hidden_blocks",
                    nn.Sequential(
                        OrderedDict([
                            *[(
                                f"hidden_block_{i}",
                                LinearBlock(
                                    hidden_sizes[i],
                                    hidden_sizes[i+1],
                                    hidden_use_dropout,
                                    hidden_dropout_rate,
                                    hidden_use_normalization,
                                )
                            )
                        for i in range(len(hidden_sizes) - 1)]
                        ])
                    ),
                ),
                (
                    "classification_head",
                    nn.Linear(
                        hidden_sizes[-1],
                        n_classes,
                    ),
                ),
            ])
        )

    @property
    def backbone(self) -> PreTrainedModel:
        return self.__backbone

    @property
    def head(self) -> nn.Sequential:
        return self.__head

    def forward(self, tokenizer_output: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.__mode == "transfer learning":
            with torch.no_grad():
                backbone_output: transformers.MaskedLMOutput = self.__backbone(
                    **tokenizer_output,
                    output_hidden_states=True,
                )
        elif self.__mode == "fine tuning":
            backbone_output = self.__backbone(
                **tokenizer_output,
                output_hidden_states=True,
            )

        if self.__aggregation == "concatenation":
            embeddings: torch.Tensor = torch.cat(
                backbone_output.hidden_states[-self.__n_last_layers:],
                dim=-1,
            )[:, 0, :]
        elif self.__aggregation == "addition":
            embeddings = torch.stack(
                backbone_output.hidden_states[-self.__n_last_layers:],
                dim=-1,
            ).sum(dim=-1)[:, 0, :]

        return self.__head(embeddings)  # type: ignore
