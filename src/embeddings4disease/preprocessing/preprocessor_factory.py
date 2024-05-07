import os
import pathlib
import random
import typing as tp
from abc import ABC, abstractmethod
from datetime import datetime
from dateutil import parser

import numpy as np
import pandas as pd
from tqdm import tqdm, std

from embeddings4disease.utils import utils


class PreprocessorFactory(ABC):
    """
    Base class for preprocessor.
    """
    def __init__(self, config: dict[str, tp.Any]):
        self.config: dict[str, tp.Any] = config

    @abstractmethod
    def create_vocab(self) -> None:
        """
        This method must create a vocab file: just a txt-file where all the tokens written line by line.
        For example:

        A11
        A22
        A33

        There is no need to add special tokens at this file, cause they are explicitly specified
        in the tokenizer's constructor.
        """
        pass

    @abstractmethod
    def create_mlm_dataset(self) -> None:
        """
        This method is used to create a dataset for Masked Language Modeling (MLM). We'll use it to train backbone model.
        It must write one transaction on one line. For example, it a person has such transactions: A11 A22, B11 B22 B33;
        they must be stored as:
        A11 A22
        B11 B22 B33
        """
        pass

    @abstractmethod
    def create_seq2seq_dataset(self) -> None:
        """
        _summary_

        Returns:
            _type_: _description_
        """
        pass


PREPROCESSOR_REGISTER: dict[str, tp.Type[PreprocessorFactory]] = {}


def preprocessor(cls: tp.Type[PreprocessorFactory]) -> tp.Type[PreprocessorFactory]:
    PREPROCESSOR_REGISTER[cls.__name__[:-19]] = cls
    return cls


@preprocessor
class MIMICPreprocessorFactory(PreprocessorFactory):
    """
    Preprocessor for the [MIMIC-4](https://physionet.org/content/mimiciv/2.2/) dataset.
    """
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

        storage_dir: pathlib.Path = pathlib.Path(
            os.path.abspath(self.config["storage_dir"])
        )
        utils.create_dir(storage_dir)

        self.storage_dir: pathlib.Path = storage_dir

    def create_seq2seq_dataset(self) -> None:
        storage_dir: pathlib.Path = self.storage_dir.joinpath(pathlib.Path("seq2seq"))
        utils.create_dir(storage_dir)

        diagnoses: pd.DataFrame = pd.read_csv(os.path.abspath(self.config["diagnoses"]))
        diagnoses = diagnoses[diagnoses.icd_version == 10]
        diagnoses = self._preprocess_codes(
            diagnoses,
            self.config["code_length"],
            self.config["code_lower_bound"],
            self.config["code_upper_bound"],
        )

        admissions: pd.DataFrame = pd.read_csv(os.path.abspath(self.config["admissions"]))
        admissions.admittime = utils.str2datetime(admissions.admittime)

        threshold_date: datetime = parser.parse(self.config["threshold_date"])

        admissions_train: pd.DataFrame = admissions[admissions.admittime < threshold_date]
        admissions_validation: pd.DataFrame = admissions[admissions.admittime >= threshold_date]

        del admissions

        train_dataframe: pd.DataFrame = diagnoses.merge(admissions_train, on=["hadm_id", "subject_id"])
        validation_dataframe: pd.DataFrame = diagnoses.merge(admissions_validation, on=["hadm_id", "subject_id"])

        del diagnoses

        self._seq2seq_dataframe2file(storage_dir.joinpath("train_transactions.txt"),
                                     train_dataframe,
                                     admissions_train,
                                     "train",
                                    )

        self._seq2seq_dataframe2file(storage_dir.joinpath("validation_transactions.txt"),
                                     validation_dataframe,
                                     admissions_validation,
                                     "validation",
                                    )

    def create_mlm_dataset(self) -> None:
        storage_dir: pathlib.Path = self.storage_dir.joinpath(pathlib.Path("mlm"))
        utils.create_dir(storage_dir)

        diagnoses: pd.DataFrame = pd.read_csv(os.path.abspath(self.config["diagnoses"]))
        diagnoses = diagnoses[diagnoses.icd_version == 10]
        diagnoses = self._preprocess_codes(
            diagnoses,
            self.config["code_length"],
            self.config["code_lower_bound"],
            self.config["code_upper_bound"],
        )

        admissions: pd.DataFrame = pd.read_csv(os.path.abspath(self.config["admissions"]))
        admissions.admittime = utils.str2datetime(admissions.admittime)

        threshold_date: datetime = parser.parse(self.config["threshold_date"])

        admissions_train: pd.DataFrame = admissions[admissions.admittime < threshold_date]
        admissions_validation: pd.DataFrame = admissions[admissions.admittime >= threshold_date]

        del admissions

        train_dataframe: pd.DataFrame = diagnoses.merge(admissions_train, on=["hadm_id", "subject_id"])
        validation_dataframe: pd.DataFrame = diagnoses.merge(admissions_validation, on=["hadm_id", "subject_id"])

        del diagnoses

        self._mlm_dataframe2file(storage_dir.joinpath("train_transactions.txt"),
                                 train_dataframe,
                                 "train",
                                )

        self._mlm_dataframe2file(storage_dir.joinpath("validation_transactions.txt"),
                                 validation_dataframe,
                                 "validation",
                                )

    def create_vocab(self) -> None:
        vocab: set[str] = set()

        with (
            open(self.storage_dir.joinpath(pathlib.Path("mlm").joinpath("train_transactions.txt")), "r") as train_file,
            open(self.storage_dir.joinpath("vocab.txt"), "w") as vocab_file,
        ):
            progress_bar: std.tqdm = tqdm(train_file)
            progress_bar.set_description("Creating vocabulary file")
            for transaction in progress_bar:
                tokens: list[str] = transaction.split()

                for token in tokens:
                    if token not in vocab:
                        vocab.add(token)
                        vocab_file.write(token + "\n")

    def _seq2seq_dataframe2file(self,
                                path: str | pathlib.Path,
                                dataframe: pd.DataFrame,
                                admissions: pd.DataFrame,
                                label: str,
                               ) -> None:
        lower_bound: int = self.config["lower_bound"]
        upper_bound: int = self.config["upper_bound"]

        with open(path, "w") as file:
            subject_ids: list[int] = list(set(dataframe["subject_id"]))

            progress_bar: std.tqdm = tqdm(subject_ids)
            progress_bar.set_description(f"Creating seq2seq {label} dataset")

            for subject_id in progress_bar:
                subject_transactions: pd.DataFrame = dataframe[dataframe["subject_id"] == subject_id]
                subject_transactions = subject_transactions.sort_values(by="admittime")

                # collect unique hadm_id, saving an order.
                unique_hadm_ids: utils.CustomOrderedSet = utils.CustomOrderedSet()
                for hadm_id in subject_transactions.hadm_id.values:
                    unique_hadm_ids.add(hadm_id)

                hadm_ids: list[int] = [hadm_id for hadm_id in unique_hadm_ids]

                if len(hadm_ids) > 1:
                    for i in range(0, len(hadm_ids) - 1):
                        source_hadm_id = hadm_ids[i]
                        source_admittime = admissions[admissions.hadm_id == source_hadm_id].admittime.values[0]

                        # collect admissions that are within the required date range.
                        target_hadm_ids = []
                        for target_hadm_id in hadm_ids[i+1:]:
                            target_admittime = admissions[admissions.hadm_id == target_hadm_id].admittime.values[0]

                            diff_in_months = ((target_admittime - source_admittime).astype("timedelta64[M]"))/np.timedelta64(1, "M")
                            if lower_bound <= diff_in_months <= upper_bound:
                                target_hadm_ids.append(target_hadm_id)

                        if len(target_hadm_ids) != 0:
                            # make target
                            source_diseases: set[str] = set(subject_transactions[subject_transactions.hadm_id == source_hadm_id].icd_code.values)

                            target_diseases: set[str] = source_diseases.copy()
                            for target_hadm_id in target_hadm_ids:
                                target_diseases.update(set(subject_transactions[subject_transactions.hadm_id == target_hadm_id].icd_code.values))
                            target_diseases.difference_update(source_diseases)

                            file.write(" ".join(source_diseases) + "," + " ".join(target_diseases) + "\n")

    def _mlm_dataframe2file(self,
                            path: str | pathlib.Path,
                            dataframe: pd.DataFrame,
                            label: str,
                           ) -> None:
        """
        Converts preprocessed pandas Dataframe into the txt file.

        Args:
            path (str | pathlib.Path): path to file where the data will be stored.
            dataframe (pd.DataFrame): dataframe you want to convert.
            label (str): label will be used as a tqdm bar description. By design, it can be either 'trainin' or 'validation'.
        """
        with open(path, "w") as file:
            subject_ids: list[int] = list(set(dataframe["subject_id"]))

            progress_bar: std.tqdm = tqdm(subject_ids)
            progress_bar.set_description(f"Creating mlm {label} dataset")
            for subject_id in progress_bar:
                subject_transactions: pd.DataFrame = dataframe[dataframe["subject_id"] == subject_id]

                unique_hadm_ids: utils.CustomOrderedSet = utils.CustomOrderedSet()
                for hadm_id in list(subject_transactions["hadm_id"]):
                    unique_hadm_ids.add(hadm_id)

                hadm_ids: list[int] = [hadm_id for hadm_id in unique_hadm_ids]

                for hadm_id in hadm_ids:
                    unique_tokens = self._get_unique_tokens(subject_transactions, hadm_id)
                    file.write(" ".join(unique_tokens) + "\n")

    @staticmethod
    def _get_unique_tokens(
        subject_transactions: pd.DataFrame, hadm_id: int
    ) -> list[str]:
        """
        Returns unique diseases that were detected during the given hadm_id with respect to order.

        Args:
            subject_transactions (pd.DataFrame): subject's datected disease.
            hadm_id (int): current hadm id.

        Returns:
            list[str]: list of unique detected diseases.
        """
        transaction: pd.DataFrame = subject_transactions[
            subject_transactions["hadm_id"] == hadm_id
        ]
        tokens: list[str] = list(transaction["icd_code"])

        custom_ordered_set: utils.CustomOrderedSet = utils.CustomOrderedSet()
        for current_token in tokens:
            custom_ordered_set.add(current_token)

        return [token for token in custom_ordered_set]

    @staticmethod
    def _preprocess_codes(
        data: pd.DataFrame,
        code_length: int,
        lower_bound: str,
        upper_bound: str,
    ) -> pd.DataFrame:
        """
        Removes some codes and cuts icd-10 code to the first code_length symbols.

        Args:
            data (pd.DataFrame): data you want to preprocess.
            code_length (int): maximum length of icd code.
            lower_bound (str): lower bound of allowed codes (including)
            upper_bound (str): upper bound of allowed codes (including)

        Returns:
            pd.DataFrame: preprocessed dataframe.
        """
        code_cutter: utils.CodeCutter = utils.CodeCutter(code_length)

        data["icd_code"] = pd.Series(data["icd_code"], dtype="string")
        data["icd_code"] = data["icd_code"].apply(code_cutter)

        data_filtered: pd.DataFrame = data[
            (lower_bound <= data["icd_code"]) & (data["icd_code"] <= upper_bound)
        ]

        return data_filtered


@preprocessor
class SecretDatasetPreprocessorFactory(PreprocessorFactory):
    """
    Preprocessor for the secret dataset.
    """
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__(config)

        random_seed: int = self.config["random_seed"]
        random.seed(random_seed)
        np.random.seed(random_seed)

        storage_dir: pathlib.Path = pathlib.Path(
            os.path.abspath(self.config["storage_dir"])
        )
        utils.create_dir(storage_dir)

        self.storage_dir: pathlib.Path = storage_dir

    def make_train_val_split(self) -> None:
        diagnoses: pd.DataFrame = pd.read_csv(
            pathlib.Path(os.path.abspath(self.config["data"]))
        )

        diagnoses = self._preprocess_column(
            diagnoses,
            "icd_code",
            self.config["code_length"],
            self.config["code_lower_bound"],
            self.config["code_upper_bound"],
        )

        member_ids: list[int] = list(set(diagnoses["member_id"]))

        with (
            open(self.storage_dir.joinpath("train_transactions_single.txt"), "w") as train_single,
            open(self.storage_dir.joinpath("train_transactions_pair.txt"), "w") as train_pair,
            open(self.storage_dir.joinpath("val_transactions_single.txt"), "w") as val_single,
            open(self.storage_dir.joinpath("val_transactions_pair.txt"), "w") as val_pair,
        ):
            for member_index in tqdm(range(len(member_ids))):
                member_transactions: pd.DataFrame = diagnoses[
                    diagnoses["member_id"] == member_ids[member_index]
                ]

                unique_dates_of_service: utils.CustomOrderedSet = (
                    utils.CustomOrderedSet()
                )
                for date in list(member_transactions["date_of_service"]):
                    unique_dates_of_service.add(date)
                dates_of_service: list[int] = [date for date in unique_dates_of_service]

                if len(dates_of_service) == 1:
                    # always send to training
                    unique_tokens = self._get_unique_tokens(
                        member_transactions, dates_of_service[0]
                    )
                    train_single.write(" ".join(unique_tokens) + "\n")
                else:
                    to_validation: bool = (
                        True if np.random.uniform() < self.config["epsilon"] else False
                    )

                    dates_of_service_subset: list[int] = dates_of_service[:-2]
                    for date_index in range(len(dates_of_service_subset)):
                        unique_tokens = self._get_unique_tokens(
                            member_transactions, dates_of_service_subset[date_index]
                        )
                        train_single.write(" ".join(unique_tokens) + "\n")

                        if date_index < len(dates_of_service_subset) - 1:
                            next_unique_tokens = self._get_unique_tokens(
                                member_transactions,
                                dates_of_service_subset[date_index + 1],
                            )
                            train_pair.write(
                                " ".join(unique_tokens) + "," + " ".join(next_unique_tokens) + "\n"
                            )

                    if to_validation:
                        # send the last 2 transactions to validation
                        dates_of_service_subset = dates_of_service[-2:]
                        for date_index in range(len(dates_of_service_subset)):
                            unique_tokens = self._get_unique_tokens(
                                member_transactions, dates_of_service_subset[date_index]
                            )
                            val_single.write(" ".join(unique_tokens) + "\n")

                            if date_index < len(dates_of_service_subset) - 1:
                                next_unique_tokens = self._get_unique_tokens(
                                    member_transactions,
                                    dates_of_service_subset[date_index + 1],
                                )
                                val_pair.write(
                                    " ".join(unique_tokens) + "," + " ".join(next_unique_tokens) + "\n"
                                )
                    else:
                        # send the last 2 transactions to training
                        dates_of_service_subset = dates_of_service[-2:]
                        for date_index in range(len(dates_of_service_subset)):
                            unique_tokens = self._get_unique_tokens(
                                member_transactions, dates_of_service_subset[date_index]
                            )
                            train_single.write(" ".join(unique_tokens) + "\n")

                            if date_index < len(dates_of_service_subset) - 1:
                                next_unique_tokens = self._get_unique_tokens(
                                    member_transactions,
                                    dates_of_service_subset[date_index + 1],
                                )
                                train_pair.write(
                                    " ".join(unique_tokens) + "," + " ".join(next_unique_tokens) + "\n"
                                )

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
    def _get_unique_tokens(subject_transactions: pd.DataFrame, date: int) -> list[str]:
        transaction: pd.DataFrame = subject_transactions[
            subject_transactions["date_of_service"] == date
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
