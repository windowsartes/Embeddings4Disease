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
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 use_dropout: bool = True,
                 dropout_rate: float=0.1,
                 use_normalization: bool = True,
                ):
        super().__init__()

        self.block = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(input_size, output_size, bias=False)),
            ("normalization", nn.BatchNorm1d(output_size) if use_normalization else nn.Identity()),
            ("activation", nn.ReLU()),
            ("dropout", nn.Dropout1d(dropout_rate) if use_dropout else nn.Identity()),
        ]))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)  # type: ignore


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
    def __init__(self,
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

        self.backbone: PreTrainedModel = backbone
        self.n_classes: int = n_classes
        self.n_last_layers: int = min(n_last_layers, len(self.backbone.bert.encoder.layer))

        if mode not in ["transfer learning", "fine tuning"]:
            warnings.warn(f"{mode} doen't supported; 'transfer learning' mode will be used",
                          UserWarning)
            self.mode = "transfer learning"
        else:
            self.mode = mode

        if self.mode == "transfer learning":
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        elif self.mode == "fine tuning":
            self.backbone.train()

        if aggregation not in ["concatenation", "addition"]:
            warnings.warn(f"{aggregation} doen't supported; 'concatenation' aggregation will be used",
                          UserWarning)
            self.aggregation = "concatenation"
        else:
            self.aggregation = aggregation

        embedding_size: int = self.backbone.bert.embeddings.word_embeddings.embedding_dim

        if self.aggregation == "concatenation":
            input_size = embedding_size * self.n_last_layers
        elif self.aggregation == "addition":
            input_size = embedding_size

        self.head = nn.Sequential(OrderedDict([
            (
                "input_block",
                LinearBlock(input_size, hidden_sizes[0])
            ),
            (
                "hidden_blocks",
                nn.Sequential(OrderedDict([
                    *[(f"hidden_block_{i}",
                        LinearBlock(
                            hidden_sizes[i],
                            hidden_sizes[i+1],
                            hidden_use_dropout,
                            hidden_dropout_rate,
                            hidden_use_normalization
                        )
                    )
                    for i in range(len(hidden_sizes) - 1)]
                ])),
            ),
            (
                "classification_head",
                nn.Linear(hidden_sizes[-1], n_classes),
            ),
        ]))

    def forward(self, tokenizer_output: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "transfer learning":
            with torch.no_grad():
                backbone_output: transformers.MaskedLMOutput = self.backbone(**tokenizer_output,
                                                                             output_hidden_states=True
                                                                            )
        elif self.mode == "fine tuning":
            backbone_output = self.backbone(**tokenizer_output, output_hidden_states=True)

        if self.aggregation == "concatenation":
            embeddings: torch.Tensor = torch.cat(backbone_output.hidden_states[-self.n_last_layers:],
                                                 dim = -1,
                                                )[:, 0, :]
        elif self.aggregation == "addition":
            embeddings = torch.stack(backbone_output.hidden_states[-self.n_last_layers:],
                                     dim=-1
                                    ).sum(dim=-1)[:, 0, :]

        return self.head(embeddings)  # type: ignore
