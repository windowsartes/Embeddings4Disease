import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class MaskingCollator:
    """
    This class is used as a collate_fn function inside dataloder in MetricComputerValidationCallback.
    It just takes a string with tokens, [MASK]ing last token, applyes tokenizer to the output then can be
    passed throw BERT-like model; also it returns true token that was [MASK]ed so you can validate your model.

    In __init__ this collator receives following arguments:
    Args:
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): tokenizer you want to use on your data.
        seq_len (int): maximum length in tokens one input sequence can be.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, seq_len: int
    ):
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer
        self.seq_len: int = seq_len

    def __call__(
        self, sequence_batch: list[str]
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """
        This method will be hiddenly used by a dataloder if an instance of this class will be used as a
        collate_fn function. It receives a list of string (with tokens separated by space), masked the
        last token and applyes the tokenizer.

        Args:
            sequence_batch (list[str]): list of string with tokens separated by space.

        Returns:
            tuple[dict[str, torch.Tensor], list[str]]: preprocessed list of string and true tokens that were masked.
        """

        inputs_sequence: list[str] = []
        answers: list[str] = []

        for sequence in sequence_batch:
            sequence_splitted = sequence.split(" ")

            sequence_splitted = sequence_splitted[:seq_len]

            inputs_sequence.append(
                " ".join(sequence_splitted[:-1] + [tokenizer.mask_token])
            )
            answers.append(sequence_splitted[-1])

        inputs: dict[str, torch.Tensor] = self.tokenizer(
            inputs_sequence, padding="longest", return_tensors="pt"
        )

        return inputs, answers
