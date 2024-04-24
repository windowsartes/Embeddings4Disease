import pytest
import torch
import torch.nn as nn

from embeddings4disease.head.architectures.multilabel_head import LinearBlock


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
    False, True
]

@pytest.mark.parametrize("use_normalization", test_data)
def test_linear_block_use_normalization(use_normalization: bool):
    linear_block: LinearBlock = LinearBlock(4, 16, use_normalization=use_normalization)

    assert isinstance(linear_block.block.normalization, nn.BatchNorm1d) == use_normalization


test_data = [
    False, True
]

@pytest.mark.parametrize("use_dropout", test_data)
def test_linear_block_use_normalization(use_dropout: bool):
    linear_block: LinearBlock = LinearBlock(4, 16, use_dropout=use_dropout)

    assert isinstance(linear_block.block.dropout, nn.Dropout1d) == use_dropout
