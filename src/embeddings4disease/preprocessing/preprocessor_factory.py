import os
import pathlib
import random
import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm

from embeddings4disease.utils import utils


class PreprocessorFactory(ABC):
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    @abstractmethod
    def make_train_val_split(self) -> None:
        """
        This method must make a train-validation split in special format: it must create 2 files
        for trainin and 2 files for validation.
        At the first file, transaction are written line-by-line. For example:

        A11 A22 A33
        B11 B22

        At the second file, we create a seq2seq dataset. So the format must be: source transaction at one line,
        target transaction at the next line, than blank line. For example:

        A11, A22
        A22, A33

        B11, B22, B33
        B22, B44, B55

        Since we haven't done seq2seq yet, this format may change.
        """
        pass

    @abstractmethod
    def create_vocab(self) -> None:
        """
        This method must create a vocab file: just a txt-file where all the tokens written line by line.
        For example:

        A11
        A22
        A33

        Also you may or may not add special token at this file, cause they are explicitly specified in the constructor.
        """
        pass


CLASS_REGISTER: dict[str, tp.Type[PreprocessorFactory]] = {}


def preprocessor(cls: tp.Type[PreprocessorFactory]) -> tp.Type[PreprocessorFactory]:
    CLASS_REGISTER[cls.__name__[:-19]] = cls
    return cls


@preprocessor
class MIMICPreprocessorFactory(PreprocessorFactory):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

        random_seed: int = self.config["random_seed"]
        random.seed(random_seed)
        np.random.seed(random_seed)

        storage_dir: pathlib.Path = pathlib.Path(os.path.abspath(self.config["storage_dir"]))
        utils.create_dir(storage_dir)

        self.storage_dir: pathlib.Path = storage_dir

    def make_train_val_split(self) -> None:
        diagnoses: pd.DataFrame = pd.read_csv(
            pathlib.Path(os.path.abspath(self.config["data"]))
        )
        diagnoses = diagnoses[diagnoses.icd_version == 10]
        diagnoses = self._preprocess_column(
            diagnoses,
            "icd_code",
            self.config["code_length"],
            self.config["code_lower_bound"],
            self.config["code_upper_bound"],
        )

        subject_ids: list[int] = list(set(diagnoses["subject_id"]))

        first_write_train: bool = True
        first_write_val: bool = True

        with (
            open(self.storage_dir.joinpath("train_transactions_single.txt"), "w") as train_single,
            open(self.storage_dir.joinpath("train_transactions_pair.txt"), "w") as train_pair,
            open(self.storage_dir.joinpath("val_transactions_single.txt"), "w") as val_single,
            open(self.storage_dir.joinpath("val_transactions_pair.txt"), "w") as val_pair,
        ):
            for subject_index in tqdm(range(len(subject_ids))):
                subject_transactions: pd.DataFrame = diagnoses[
                    diagnoses["subject_id"] == subject_ids[subject_index]
                ]

                unique_hadm_ids: utils.CustomOrderedSet = utils.CustomOrderedSet()
                for hadm_id in list(subject_transactions["hadm_id"]):
                    unique_hadm_ids.add(hadm_id)
                hadm_ids: list[int] = [hadm_id for hadm_id in unique_hadm_ids]

                if len(hadm_ids) == 1:
                    # оcтавляем для обучения всегда
                    unique_tokens = self._get_unique_tokens(
                        subject_transactions, hadm_ids[0]
                    )
                    train_single.write(" ".join(unique_tokens) + "\n")
                else:
                    to_validation: bool = (
                        True if np.random.uniform() < self.config["epsilon"] else False
                    )

                    hadm_id_subset: list[int] = hadm_ids[:-2]
                    for hadm_id_index in range(len(hadm_id_subset)):
                        unique_tokens = self._get_unique_tokens(
                            subject_transactions, hadm_id_subset[hadm_id_index]
                        )
                        train_single.write(" ".join(unique_tokens) + "\n")

                        if hadm_id_index < len(hadm_id_subset) - 1:
                            next_unique_tokens = self._get_unique_tokens(
                                subject_transactions, hadm_id_subset[hadm_id_index + 1]
                            )
                            train_pair.write(" ".join(unique_tokens) + "," + " ".join(next_unique_tokens) + "\n")

                    if to_validation:
                         # отправляем последние 2 транзакции в валидацию
                        hadm_id_subset = hadm_ids[-2:]
                        for hadm_id_index in range(len(hadm_id_subset)):
                            unique_tokens = self._get_unique_tokens(
                                subject_transactions, hadm_id_subset[hadm_id_index]
                            )
                            val_single.write(" ".join(unique_tokens) + "\n")

                            if hadm_id_index < len(hadm_id_subset) - 1:
                                next_unique_tokens = self._get_unique_tokens(
                                    subject_transactions, hadm_id_subset[hadm_id_index + 1]
                                )
                                val_pair.write(" ".join(unique_tokens) + "," + " ".join(next_unique_tokens) + "\n")
                    else:
                        # отправляем последние 2 транзакции в обучение
                        hadm_id_subset = hadm_ids[-2:]
                        for hadm_id_index in range(len(hadm_id_subset)):
                            unique_tokens = self._get_unique_tokens(
                                subject_transactions, hadm_id_subset[hadm_id_index]
                            )
                            train_single.write(" ".join(unique_tokens) + "\n")

                            if hadm_id_index < len(hadm_id_subset) - 1:
                                next_unique_tokens = self._get_unique_tokens(
                                    subject_transactions, hadm_id_subset[hadm_id_index + 1]
                                )
                                train_pair.write(" ".join(unique_tokens) + "," + " ".join(next_unique_tokens) + "\n")
                                

    def create_vocab(self) -> None:
        vocab: set[str] = set()

        with (
            open(self.storage_dir.joinpath("vocab.txt"), "w") as vocab_file,
            open(self.storage_dir.joinpath("train_transactions_single.txt"), "r") as train_single,
        ):
            for transaction in train_single:
                tokens: list[str] = transaction.split()

                for token in tokens:
                    if token not in vocab:
                        vocab.add(token)
                        vocab_file.write(token + "\n")

    @staticmethod
    def _get_unique_tokens(
        subject_transactions: pd.DataFrame, hadm_id: int
    ) -> list[str]:
        transaction: pd.DataFrame = subject_transactions[
            subject_transactions["hadm_id"] == hadm_id
        ]
        tokens: list[str] = list(transaction["icd_code"])

        custom_ordered_set: utils.CustomOrderedSet = utils.CustomOrderedSet()
        for current_token in tokens:
            custom_ordered_set.add(current_token)

        return [token for token in custom_ordered_set]

    @staticmethod
    def _preprocess_column(
        data: pd.DataFrame,
        target_column: str,
        code_length: int,
        lower_bound: str,
        upper_bound: str,
    ) -> pd.DataFrame:
        code_cutter: utils.CodeCutter = utils.CodeCutter(code_length)

        data[target_column] = pd.Series(data[target_column], dtype="string")
        data[target_column] = data[target_column].apply(code_cutter)

        data_filtered: pd.DataFrame = data[
            (lower_bound <= data["icd_code"]) & (data["icd_code"] <= upper_bound)
        ]

        return data_filtered
