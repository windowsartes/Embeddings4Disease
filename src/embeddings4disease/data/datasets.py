import os
import pathlib
import pickle
import random
from filelock import FileLock

import torch
import transformers
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class CustomLineByLineDataset(Dataset):
    """
    This simple class is commonly used inside MetricComputerValidationCallback.
    It just reads the data from file and store it inside.

    Args:
        path (str): path to data you want to use.
    """

    def __init__(self, path: str | pathlib.Path):
        super().__init__()

        with open(path, "r") as input_file:
            self.content: list[str] = input_file.readlines()

    def __len__(self) -> int:
        """
        Return the number of stored transactions.

        Returns:
            int: the number of stored transactions
        """
        return len(self.content)

    def __getitem__(self, index: int) -> str:
        """
        Strip and return transaction by its index.

        Args:
            index (int): index of a transaction.

        Returns:
            str: stripped transaction.
        """
        return self.content[index].strip()


class CustomTextDatasetForNextSentencePrediction(Dataset):
    """
    This Dataset is totally based on transformer' TextDatasetForNextSentencePrediction but with some updates.
    It is used for training BERT with NSP, but currently we don't support this kind of models.

    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        seq_len: int,
        file_path: str,
        block_size: int,
        overwrite_cache: bool = False,
        short_seq_probability: float = 0.1,
        nsp_probability: float = 0.5,
    ):
        super().__init__()

        if not os.path.isfile(file_path):
            raise ValueError(f"Input file path {file_path} not found")

        self.seq_len: int = seq_len
        self.short_seq_probability: float = short_seq_probability
        self.nsp_probability: float = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_nsp_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path: str = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                with open(cached_features_file, "rb") as handle:
                    self.examples: list[dict[str, torch.Tensor]] = pickle.load(handle)

            else:
                self.documents: list[list[list[int]]] = [[]]
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line: str = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens: list[int] = tokenizer.convert_tokens_to_ids(
                            tokenizer.tokenize(line)
                        )
                        if tokens:
                            self.documents[-1].append(tokens)

                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index, block_size)

                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_examples_from_document(
        self, document: list[list[int]], doc_index: int, block_size: int
    ) -> None:
        """Creates examples for a single document."""

        max_num_tokens: int = block_size - self.tokenizer.num_special_tokens_to_add(
            pair=True
        )

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length: int = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk: list[list[int]] = []  # a buffer stored current working segments
        current_length: int = 0
        i: int = 0

        while i < len(document):
            segment: list[int] = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end: int = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a: list[int] = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b: list[int] = []

                    if (
                        len(current_chunk) == 1
                        or random.random() < self.nsp_probability
                    ):
                        is_random_next: bool = True
                        target_b_length: int = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index: int = random.randint(
                                0, len(self.documents) - 1
                            )
                            if random_document_index != doc_index:
                                break

                        random_document: list[list[int]] = self.documents[
                            random_document_index
                        ]
                        random_start: int = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments: int = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    if not (len(tokens_a) >= 1):
                        raise ValueError(
                            f"Length of sequence a is {len(tokens_a)} which must be no less than 1"
                        )
                    if not (len(tokens_b) >= 1):
                        raise ValueError(
                            f"Length of sequence b is {len(tokens_b)} which must be no less than 1"
                        )

                    # add special tokens
                    tokens_a = list(dict.fromkeys(tokens_a))[: self.seq_len]
                    tokens_b = list(dict.fromkeys(tokens_b))[: self.seq_len]

                    input_ids: list[int] = (
                        self.tokenizer.build_inputs_with_special_tokens(
                            tokens_a, tokens_b
                        )
                    )
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids: list[int] = (
                        self.tokenizer.create_token_type_ids_from_sequences(
                            tokens_a, tokens_b
                        )
                    )

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(
                            token_type_ids, dtype=torch.long
                        ),
                        "next_sentence_label": torch.tensor(
                            1 if is_random_next else 0, dtype=torch.long
                        ),
                    }

                    self.examples.append(example)

                current_chunk = []
                current_length = 0

            i += 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class MultiLabelHeadDataset(Dataset):
    """
    This dataset is used to train multilabel head model.

    Args:
        path_to_data (str): path to data you want to use. Must be in seq-to-seq format:
        sorce and target sequences at the same line separated by comma.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): tokenizer you want to use.
        predict_unk (bool): can <UNK> presents in the target vector or not.

    """

    def __init__(
        self,
        path_to_data: pathlib.Path | str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        predict_unk: bool = False,
    ):
        super().__init__()

        with open(path_to_data, "r") as input_file:
            self.data = input_file.readlines()

        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer

        # technical tokens are also included here, but if we remove them,
        # we'll have to change the indices after the model predictor,
        # and this will be quite inconvenient, but overall ok.
        self.num_classes: int = tokenizer.vocab_size

        # The idea is that UNK can be understood as some kind of disease that is not in the dictionary.
        # Then it may be logical to predict it.
        self.predict_unk: bool = predict_unk

    def __len__(self) -> int:
        """
        Returns lengths of stored dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        """
        This method will return you a tuple with source sequence as a string and
        target tokens indexes as a torch.Tensor one-hot vector.

        Args:
            index (int): index of a pair of transactions.

        Returns:
            tuple[str, torch.LongTensor]: sorce seq. as a string, new target seq. tokens as a one-hot vector.
        """
        source_seq_str, target_seq_str = self.data[index].split(",")

        source_seq: list[str] = self.tokenizer.tokenize(source_seq_str)
        target_seq: list[str] = self.tokenizer.tokenize(target_seq_str)

        if self.predict_unk:
            target_tokens: str = " ".join(
                [token for token in target_seq if token not in source_seq]
            )
        else:
            target_tokens = " ".join([
                    token
                    for token in target_seq
                    if token not in source_seq
                    and token in self.tokenizer.get_vocab()
                    and token not in self.tokenizer.special_tokens_map
                ]
            )

        target_tokens_codes: list[int] = self.tokenizer.encode(target_tokens)[1:-1]

        target_tokens_codes_tensor: torch.Tensor = torch.tensor(target_tokens_codes).long()

        target_one_hot: torch.Tensor = torch.nn.functional.one_hot(
            target_tokens_codes_tensor, num_classes=self.num_classes
        )
        target_one_hot = target_one_hot.sum(dim=0).float()

        return source_seq_str, target_one_hot


class EncoderDecoderDataset(Dataset):
    def __init__(self,
                 path: pathlib.Path | str,
                 tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                 max_length: int):
        super().__init__()

        self._tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer
        self._max_length: int = max_length

        with open(path, "r") as input_file:
            self.pairs: list[str] = input_file.readlines()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> transformers.BatchEncoding:
        source_str, target_str = self.pairs[index].strip().split(",")

        model_inputs = self._tokenizer(
            source_str,
            text_target=target_str,
            max_length=self._max_length,
            truncation=True,
        )

        return model_inputs


class SourceTargetStringsDataset(Dataset):
    def __init__(self, path: pathlib.Path | str):
        with open(path, "r") as input_file:
            self.pairs: list[str] = input_file.readlines()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        source_str, target_str = self.pairs[index].strip().split(",")

        return source_str, target_str
