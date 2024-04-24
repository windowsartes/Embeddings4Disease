from collections import OrderedDict

import torch
from torch import nn
from transformers import PreTrainedModel


class LinearBlock(nn.Module):
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
        print(input_tensor)
        return self.block(input_tensor)


class MultiLabelHead(nn.Module):
    def __init__(self,
                 backbone: PreTrainedModel,
                 n_classes: int,
                 n_last_layers: int = 4,
                 aggregation = "concatenation",
                 mode = "transfer learning",
                 hidden_sizes: tuple[int] = (1024, 512, 256, 128),
                 hidden_use_dropout: bool = True,
                 hidden_dropout_rate: float = 0.1,
                 hidden_use_normalization: bool = True
                ):
        super().__init__()

        self.backbone: PreTrainedModel = backbone
        self.n_classes: int = n_classes
        self.n_last_layers: int = min(n_last_layers, len(model.backbone.bert.encoder.layer))

        if mode not in ["transfer learning", "fine tuning"]:
            warnings.warn(f"{mode} doen't supported; 'transfer learning' mode will be used",
                          UserWarning)
            self.mode = "transfer learning"
        else:
            self.mode: str = mode

        if self.mode == "transfer learning":
            self.backbone.eval()
        elif self.mode == "fine tuning":
            self.backbone.train()

        if aggregation not in ["concatenation", "addition"]:
            warnings.warn(f"{aggregation} doen't supported; 'concatenation' aggregation will be used",
                          UserWarning)
            self.aggregation = "concatenation"
        else:
            self.aggregation: str = aggregation

        embedding_size: int = self.backbone.bert.embeddings.word_embeddings.embedding_dim

        if aggregation == "concatenation":
            input_size = embedding_size * self.n_last_layers
        elif aggregation == "addition":
            input_size = embedding_size

        self.input_block: LinearBlock = LinearBlock(input_size, hidden_sizes[0])

        self.hidden_blocks: nn.Sequential = nn.Sequential(OrderedDict([
            *[(f"hidden_block_{i}",
               LinearBlock(hidden_sizes[i], hidden_sizes[i+1], use_dropout,
                           dropout_rate, use_normalization))
             for i in range(len(hidden_sizes) - 1)]
        ]))

        self.classification_head: nn.Linear = nn.Linear(hidden_sizes[-1], n_classes)

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

        return self.classification_head(self.hidden_blocks(self.input_block(embeddings)))
    