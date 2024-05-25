import pytest
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer

from embeddings4disease.head.architectures.multilabel_head import LinearBlock, MultiLabelHead


test_data = [
    (1, 1), (1, 2), (2, 8), (16, 32), (17, 19), (384*4, 911),
]

@pytest.mark.parametrize("input_size,output_size", test_data)
def test_linear_block_output_shape(input_size: int, output_size: int):
    linear_block: LinearBlock = LinearBlock(input_size, output_size)

    input_tensor: torch.Tensor = torch.ones((8, input_size))
    output_tensor: torch.Tensor = linear_block(input_tensor)

    assert output_tensor.shape == (8, output_size)


test_data = [
    False, True,
]

@pytest.mark.parametrize("use_normalization", test_data)
def test_linear_block_use_normalization(use_normalization: bool):
    linear_block: LinearBlock = LinearBlock(4, 16, use_normalization=use_normalization)

    assert isinstance(linear_block.block.normalization, nn.BatchNorm1d) == use_normalization


test_data = [
    False, True,
]

@pytest.mark.parametrize("use_dropout", test_data)
def test_linear_block_use_dropout(use_dropout: bool):
    linear_block: LinearBlock = LinearBlock(4, 16, use_dropout=use_dropout)

    assert isinstance(linear_block.block.dropout, nn.Dropout1d) == use_dropout


test_data = [
    ("transfer learning", False),
    ("fine tuning", True),
]

@pytest.mark.parametrize("mode,requires_grad", test_data)
def test_multilabel_head_requires_grad(mode: str, requires_grad: bool):
    backbone = BertForMaskedLM.from_pretrained("windowsartes/bert")
    tokenizer = BertTokenizer.from_pretrained("windowsartes/bert_tokenizer")

    multilabel_head_model = MultiLabelHead(backbone, tokenizer.vocab_size, mode=mode)

    for param in multilabel_head_model.backbone.parameters():
        assert param.requires_grad == requires_grad

    for param in multilabel_head_model.head.parameters():
        assert param.requires_grad == True


test_data = [
    ("transfer learning", False),
    ("fine tuning", True),
]

@pytest.mark.parametrize("mode,is_training", test_data)
def test_multilabel_head_training_mode(mode: str, is_training: bool):
    backbone = BertForMaskedLM.from_pretrained("windowsartes/bert")
    tokenizer = BertTokenizer.from_pretrained("windowsartes/bert_tokenizer")

    multilabel_head_model = MultiLabelHead(backbone, tokenizer.vocab_size, mode=mode)

    assert multilabel_head_model.backbone.training == is_training


test_data = [
    False, True,
]

@pytest.mark.parametrize("use_dropout", test_data)
def test_multilabel_head_hiddens_dropout_usage(use_dropout: bool):
    backbone = BertForMaskedLM.from_pretrained("windowsartes/bert")
    tokenizer = BertTokenizer.from_pretrained("windowsartes/bert_tokenizer")

    multilabel_head_model = MultiLabelHead(backbone, tokenizer.vocab_size,
                                           hidden_use_dropout=use_dropout)

    for linear_block in multilabel_head_model.head.hidden_blocks:
        assert isinstance(linear_block.block.dropout, nn.Dropout1d) == use_dropout


test_data = [
    False, True,
]

@pytest.mark.parametrize("use_normalization", test_data)
def test_multilabel_head_hiddens_normalization_usage(use_normalization: bool):
    backbone = BertForMaskedLM.from_pretrained("windowsartes/bert")
    tokenizer = BertTokenizer.from_pretrained("windowsartes/bert_tokenizer")

    multilabel_head_model = MultiLabelHead(backbone, tokenizer.vocab_size,
                                           hidden_use_normalization=use_normalization)

    for linear_block in multilabel_head_model.head.hidden_blocks:
        assert isinstance(linear_block.block.normalization, nn.BatchNorm1d) == use_normalization


test_data = [
    (1024, 512, 256, 128),
    (1024, 512, 256),
    (1024, 512),
    (1024,),
]

@pytest.mark.parametrize("hidden_sizes", test_data)
def test_multilabel_head_hidden_sizes(hidden_sizes: tuple[int]):
    backbone = BertForMaskedLM.from_pretrained("windowsartes/bert")
    tokenizer = BertTokenizer.from_pretrained("windowsartes/bert_tokenizer")

    multilabel_head_model = MultiLabelHead(backbone, tokenizer.vocab_size,
                                           hidden_sizes=hidden_sizes)

    assert len(multilabel_head_model.head.hidden_blocks) == len(hidden_sizes) - 1


def test_multilabel_head_wrong_mode():
    backbone = BertForMaskedLM.from_pretrained("windowsartes/bert")
    tokenizer = BertTokenizer.from_pretrained("windowsartes/bert_tokenizer")

    with pytest.warns(UserWarning):
        multilabel_head_model = MultiLabelHead(backbone, tokenizer.vocab_size,
                                               mode="wrong mode")


def test_multilabel_head_wrong_aggregation():
    backbone = BertForMaskedLM.from_pretrained("windowsartes/bert")
    tokenizer = BertTokenizer.from_pretrained("windowsartes/bert_tokenizer")

    with pytest.warns(UserWarning):
        multilabel_head_model = MultiLabelHead(backbone, tokenizer.vocab_size,
                                               aggregation="wrong aggregation")


test_data = [
    (1), (2), (8), (911),
]

@pytest.mark.parametrize("n_classes", test_data)
def test_multilabel_head_output_shape(n_classes: int):
    backbone = BertForMaskedLM.from_pretrained("windowsartes/bert")
    tokenizer = BertTokenizer.from_pretrained("windowsartes/bert_tokenizer")

    source_sequences: list[str] = [
                                   "A41 G20 E86 I10 E11 G30 F01 E78 K21 F02 K11 F31",
                                   "K92 E43 K70 K76 A04 I85 F17 I95 E86 K31 F10 K72 K21 I10 F41",
                                   "K70 E46 A04 K76 I10 K21 F41 F10 F17 K31",
                                   "K56 K59",
                                   "E86 I95 E04 I50 I25 J45 D64 I35 K21 M50",
                                  ]
    source_sequences_tokenized = tokenizer(source_sequences, padding="longest", return_tensors="pt",
                                            truncation=True, max_length=24)

    multilabel_head_model = MultiLabelHead(backbone, n_classes=n_classes)

    assert multilabel_head_model(source_sequences_tokenized).shape == (len(source_sequences), n_classes)
