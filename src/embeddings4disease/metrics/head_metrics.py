import math
import os
import warnings
from abc import ABC
from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.stats as ss  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
import transformers
import typing as tp
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


@dataclass
class ConfidenceInterval:
    estimation: float
    interval: tuple[float, float]


METRIC_REGISTER: dict[str, tp.Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]] = {}

def metric(function: tp.Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]) -> \
        tp.Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]:
    METRIC_REGISTER[function.__name__[9:]] = function

    return function


class MetricComputerInterface(ABC):
    """
    Base class for the metric computer. It contains the metrics we support. In the case you want to add new metric just
    implement it there and mark it with the 'metric' decorator defined above.
    """
    def __init__(self) -> None:
        self.__use_confidence_interval: bool = False
        self.__interval_type: tp.Optional[str] = None
        self.__confidence_level: float = 0.95

    def get_metrics_value(self, *args, **kwargs) -> dict[str, float] | dict[str, ConfidenceInterval]:  # type: ignore
        pass

    def __call__(self, *args, **kwargs) -> dict[str, float] | dict[str, ConfidenceInterval]:  # type: ignore
        pass

    @metric
    @staticmethod
    def _compute_exact_match(answer: np.ndarray, predicted_tokens: np.ndarray) -> float:
        """
        Computes exact match. It equals to 1 in the case answer and predicted tokens are the same else 0.

        Args:
            answer (np.ndarray): Real classes we want to predict. Multy-hot vector.
            predicted_tokens (np.ndarray): classes your model has predicted. Multy-hot vector.

        Returns:
            float: exact match value.
        """
        return float(np.all(answer == predicted_tokens))

    @metric
    @staticmethod
    def _compute_accuracy(answer: np.ndarray, predicted_tokens: np.ndarray) -> float:
        """
        Computes accuracy. Here accuracy is a ratio of truly predicted classes.

        Args:
            answer (np.ndarray): Real classes we want to predict. Multy-hot vector.
            predicted_tokens (np.ndarray): classes your model has predicted. Multy-hot vector.

        Returns:
            float: accuracy value.
        """
        eps: float = 1e-9

        return float(sum(np.logical_and(answer, predicted_tokens)) / (sum(np.logical_or(answer, predicted_tokens)) + eps))

    @metric
    @staticmethod
    def _compute_recall(answer: np.ndarray, predicted_tokens: np.ndarray) -> float:
        """
        Computes recall.

        Args:
            answer (np.ndarray): Real classes we want to predict. Multy-hot vector.
            predicted_tokens (np.ndarray): classes your model has predicted. Multy-hot vector.

        Returns:
            float: recall value.
        """
        if sum(answer) == 0:  # like in the precision case, it's 0/0
            return 1.

        return float(sum(np.logical_and(answer, predicted_tokens)) / sum(answer))

    @metric
    @staticmethod
    def _compute_precision(answer: np.ndarray, predicted_tokens: np.ndarray) -> float:
        """
        Computes precision.

        Args:
            answer (np.ndarray): Real classes we want to predict. Multy-hot vector.
            predicted_tokens (np.ndarray): classes your model has predicted. Multy-hot vector.

        Returns:
            float: precision value.
        """
        if sum(predicted_tokens) == 0:  # in this case we have an empty predicted_tokens seq. so it's 0/0.
            return 1.

        return float(sum(np.logical_and(answer, predicted_tokens)) / sum(predicted_tokens))

    @metric
    @staticmethod
    def _compute_f_score(answer: np.ndarray, predicted_tokens: np.ndarray, beta: float = 1.) -> float:
        """
        Computes f_beta-score

        Args:
            answer (np.ndarray): Real classes we want to predict. Multy-hot vector.
            predicted_tokens (np.ndarray): classes your model has predicted. Multy-hot vector.

        Returns:
            float: f-beta-score value.
        """
        eps: float = 1e-9

        precision: float = MetricComputerInterface._compute_precision(answer, predicted_tokens)
        recall: float = MetricComputerInterface._compute_recall(answer, predicted_tokens)

        return ((1 + beta**2) * precision * recall) / ((beta**2) * precision + recall + eps)

    def _get_point_estimation(
        self,
        metrics_storage: dict[str, list[float]]
    ) -> dict[str, float]:
        metrics_value: dict[str, float] = {}

        for metric in metrics_storage:
            metrics_value[metric] = float(np.mean(metrics_storage[metric]))

        return metrics_value

    def _get_confidence_interval(
        self,
        metrics_storage: dict[str, list[float]],
    ) -> dict[str, ConfidenceInterval]:
        n_resamples: int = 10000

        metrics_value: dict[str, ConfidenceInterval] = {}

        for metric in metrics_storage:
            data = metrics_storage[metric]
            data_bootstrapped = np.random.choice(data, size=(n_resamples, len(data)))

            point_estimation = np.mean(data)
            bootstrap_estimations = np.mean(data_bootstrapped, axis=1)


            if self.__interval_type == "standart":
                quantile = ss.norm.ppf((1 + self.__confidence_level) / 2, loc=0, scale=1)

                std_estimation: float = np.std(bootstrap_estimations, ddof=1)

                metrics_value[metric] = ConfidenceInterval(
                    float(round(point_estimation, 4)),
                    (
                        float(round(point_estimation - quantile * std_estimation / np.sqrt(len(data)), 4)),
                        float(round(point_estimation + quantile * std_estimation / np.sqrt(len(data)), 4))
                    ),
                )
            elif self.__interval_type == "normal":
                bootstrap_estimations_std = np.std(bootstrap_estimations, ddof=1)
                quantile = ss.norm.ppf((1 + self.__confidence_level) / 2, loc=0, scale=1)

                metrics_value[metric] = ConfidenceInterval(
                    float(round(point_estimation, 4)),
                    (
                        float(round(point_estimation - quantile * bootstrap_estimations_std, 4)),
                        float(round(point_estimation + quantile * bootstrap_estimations_std, 4))
                    ),
                )

            elif self.__interval_type == "quantile":
                bootstrap_estimations_sorted = sorted(bootstrap_estimations)

                metrics_value[metric] = ConfidenceInterval(
                    float(round(point_estimation, 4)),
                    (
                        float(round(bootstrap_estimations_sorted[math.floor(n_resamples * \
                            ((1 - self.__confidence_level) / 2))], 4)),
                        float(round(bootstrap_estimations_sorted[math.ceil(n_resamples * \
                            ((1 + self.__confidence_level) / 2))], 4)),
                    ),
                )
            else:
                bootstrap_estimations_sorted = sorted(bootstrap_estimations)

                metrics_value[metric] = ConfidenceInterval(
                    float(round(point_estimation, 4)),
                    (
                        float(round(2 * point_estimation - \
                            bootstrap_estimations_sorted[math.ceil(n_resamples * \
                                ((1 + self.__confidence_level) / 2))], 4)),
                        float(round(2 * point_estimation - \
                            bootstrap_estimations_sorted[math.floor(n_resamples * \
                                ((1 - self.__confidence_level) / 2))], 4)),
                    ),
                )

        return metrics_value


class MultiLabelHeadMetricComputer(MetricComputerInterface):
    def __init__(
        self,
        threshold: float,
        dataloader: DataLoader,
        device: torch.device,
        metrics_to_use: dict[str, bool],
        confidence_interval: bool = False,
        interval_type: tp.Optional[str] = None,
        confidence_level: float = 0.95,
    ):
        super().__init__()

        self.__threshold: float = threshold
        self.__dataloader: DataLoader = dataloader
        self.__device: torch.device = device

        self.__metrics_to_use: dict[str, bool] = metrics_to_use

        self.__use_confidence_interval: bool = confidence_interval
        self.__interval_type: tp.Optional[str] = interval_type
        self.__confidence_level: float = confidence_level

    def get_metrics_value(
        self,
        model: torch.nn.Module,
    ) -> dict[str, float] | dict[str, ConfidenceInterval]:
        """
        Base method for the model evaluation. It will computer metrics' values during the dataloader you've specified
        at the instance initialization.

        Args:
            model (PreTrainedModel): model you want to evaluate.
            metrics_to_use (dict[str, bool]): metrics you want to use during inference. In the case some metric you've
            passed in this dictionary doesn't supported, it'll be ignored.

        Returns:
            dict[str, float]: average metrics' values during evaluation.
        """
        metrics_storage: dict[str, list[float]] = {}
        for metric, usage in self.__metrics_to_use.items():
            if usage:
                if metric not in METRIC_REGISTER:
                    warnings.warn(
                        f"There is no {metric} in the supported metrics so it will be ignored",
                        SyntaxWarning
                    )
                else:
                    metrics_storage[metric] = []

        with torch.no_grad():
            progress_bar = tqdm(self.__dataloader)
            for batch_inputs, batch_answers in progress_bar:
                progress_bar.set_description("Computing metrics")

                batch_inputs = batch_inputs.to(self.__device)
                outputs = model(batch_inputs)

                probabilities: torch.Tensor = F.sigmoid(outputs)

                predictions: npt.NDArray[np.float64] = (probabilities > self.__threshold).float().detach().cpu().numpy()

                for answer, predicted_tokens in zip(batch_answers, predictions):
                    for metric in metrics_storage:
                        metrics_storage[metric].append(METRIC_REGISTER[metric](answer.numpy(), predicted_tokens))
            progress_bar.close()

        if self.__use_confidence_interval and self.__interval_type not in ["quantile", "normal", "central"]:
            warnings.warn("'interval_type' you've passed doen't supported. See docs for more details. \n" + \
                          "Only point estimation will be returned.")
            return self._get_point_estimation(metrics_storage)

        if not self.__use_confidence_interval:
            return self._get_point_estimation(metrics_storage)

        return self._get_confidence_interval(metrics_storage)


class EncoderDecoderHeadMetricComputer(MetricComputerInterface):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        metrics_to_use: dict[str, bool],
        confidence_interval: bool = False,
        interval_type: tp.Optional[str] = None,
        confidence_level: float = 0.95,
    ):
        super().__init__()

        self.__tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast = tokenizer
        self.__vocab: dict[str, int] = tokenizer.get_vocab()

        self.__metrics_to_use: dict[str, bool] = metrics_to_use

        self.__use_confidence_interval: bool = confidence_interval
        self.__interval_type: tp.Optional[str] = interval_type
        self.__confidence_level: float = confidence_level

    def __call__(
        self,
        eval_preds: transformers.trainer_utils.EvalPrediction,
    ) -> dict[str, float]:

        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.__tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.__tokenizer.pad_token_id)
        decoded_labels = self.__tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions = []
        for prediction in decoded_preds:
            if prediction != "":
                unique_tokens = list(set(prediction.split(" ")))

                prediction_tokens_codes: list[int] = [self.__vocab[token] for token in unique_tokens][1:-1]
                prediction_tokens_codes_tensor: torch.Tensor = torch.tensor(prediction_tokens_codes).long()

                prediction_one_hot: torch.Tensor = torch.nn.functional.one_hot(
                    prediction_tokens_codes_tensor, num_classes=len(self.__vocab)
                )
                prediction_one_hot = prediction_one_hot.sum(dim=0).float()
            else:
                prediction_one_hot = torch.zeros(len(self.__vocab))

            predictions.append(prediction_one_hot.numpy())

        answers = []
        for answer in decoded_labels:
            if answer != "":
                unique_tokens = list(set(answer.split(" ")))

                target_tokens_codes: list[int] = [self.__vocab[token] for token in unique_tokens][1:-1]
                target_tokens_codes_tensor: torch.Tensor = torch.tensor(target_tokens_codes).long()

                target_one_hot: torch.Tensor = torch.nn.functional.one_hot(
                    target_tokens_codes_tensor, num_classes=len(self.__vocab)
                )
                target_one_hot = target_one_hot.sum(dim=0).float()
            else:
                target_one_hot = torch.zeros(len(self.__vocab))

            answers.append(target_one_hot.numpy())

        metrics_storage: dict[str, list[float]] = {}
        for metric, usage in self.__metrics_to_use.items():
            if usage:
                if metric not in METRIC_REGISTER:
                    warnings.warn(
                        f"There is no {metric} in the supported metrics so it will be ignored",
                        SyntaxWarning
                    )
                else:
                    metrics_storage[metric] = []

        for answer, prediction in zip(answers, predictions):
            for metric in metrics_storage:
                metrics_storage[metric].append(
                    METRIC_REGISTER[metric](
                        answer,
                        prediction
                    )
                )

        if self.__use_confidence_interval and self.__interval_type not in ["standart", "quantile", "normal", "central"]:
            warnings.warn("'interval_type' you've passed doen't supported. See docs for more details. \n" + \
                            "Only point estimation will be returned.")
            point_estimations: dict[str, float] = self._get_point_estimation(metrics_storage)

            return point_estimations

        if not self.__use_confidence_interval:
            point_estimations = self._get_point_estimation(metrics_storage)

            return point_estimations

        confidence_intervals = self._get_confidence_interval(metrics_storage)

        result = {}
        for metric_name in confidence_intervals:
            result[metric_name] = confidence_intervals[metric_name].estimation
            result[f"{metric_name}_lower"] = confidence_intervals[metric_name].interval[0]
            result[f"{metric_name}_upper"] = confidence_intervals[metric_name].interval[1]

        return result


class Baseline(MetricComputerInterface):
    def __init__(
        self,
        config: dict[str, tp.Any],
    ):
        super().__init__()

        self.__config: dict[str, tp.Any] = config
        if self.__config["tokenizer"]["from_huggingface"]:
            self.__tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                self.__config["tokenizer"]["path_to_saved_tokenizer"],
            )
        else:
            self.__tokenizer = AutoTokenizer.from_pretrained(
                os.path.abspath(self.__config["tokenizer"]["path_to_saved_tokenizer"]),
            )

        self.__vocab: dict[str, int] = self.__tokenizer.get_vocab()

        counter: tp.DefaultDict[str, int] = defaultdict(int)

        with open(config["path_to_training_file"], "r") as training_file:
            for transactions_pair in training_file:
                source_str, answer_str = transactions_pair.strip().split(",")
                source_tokens = source_str.strip().split(" ")
                for token in source_tokens:
                    counter[token] += 1

        source_len2answer_lens: dict[int, dict[int, int]] = dict()

        with open(config["path_to_training_file"], "r") as training_file:
            for transactions_pair in training_file:
                source_str, answer_str = transactions_pair.strip().split(",")
                source_tokens_length: int = len(source_str.strip().split(" "))
                answer_tokens_length: int = len(answer_str.strip().split(" "))

                if source_tokens_length not in source_len2answer_lens:
                    source_len2answer_lens[source_tokens_length] = dict()

                if answer_tokens_length not in source_len2answer_lens[source_tokens_length]:
                    source_len2answer_lens[source_tokens_length][answer_tokens_length] = 0

                source_len2answer_lens[source_tokens_length][answer_tokens_length] += 1

        self.__source_len2answer_len: dict[int, int] = dict()
        for source_tokens_length in source_len2answer_lens:
            argmax: int = max(
                source_len2answer_lens[source_tokens_length],
                key=lambda k: source_len2answer_lens[source_tokens_length][k],
            )
            self.__source_len2answer_len[source_tokens_length] = argmax

        self.__use_confidence_interval: bool = self.__config["confidence_interval"]["use"]
        self.__interval_type: tp.Optional[str] = self.__config["confidence_interval"]["interval_type"]
        self.__confidence_level: float = self.__config["confidence_interval"]["confidence_level"]

        self.__counter: Counter[str] = Counter(dict(sorted(counter.items(), key=lambda item: item[1], reverse=True)))

    def get_metrics_value(
        self,
    ) -> dict[str, float] | dict[str, ConfidenceInterval]:
        metrics_storage: dict[str, list[float]] = {}
        for metric, usage in self.__config["metrics"].items():
            if usage:
                if metric not in METRIC_REGISTER:
                    warnings.warn(
                        f"There is no {metric} in the supported metrics so this key will be ignored",
                        SyntaxWarning
                    )
                else:
                    metrics_storage[metric] = []

        with open(self.__config["path_to_validation_file"], "r") as validation_file:
            progress_bar = tqdm(validation_file.readlines())
            progress_bar.set_description("Computing metrics")

            for input_sequence in progress_bar:
                source_str, answer_str = input_sequence.strip().split(",")

                if source_str != "":
                    source_unique_tokens = list(set(source_str.split(" ")))

                    source_tokens_codes: list[int] = [self.__vocab[token] for token in source_unique_tokens]

                    source_one_hot: torch.Tensor = torch.nn.functional.one_hot(
                        torch.tensor(source_tokens_codes).long(),
                        num_classes=len(self.__vocab),
                    )
                    source_one_hot = source_one_hot.sum(dim=0).float()
                else:
                    source_one_hot = torch.zeros(len(self.__vocab))

                source_len: int = int(source_one_hot.sum().item())

                if source_len == 0:
                    prediction_one_hot: torch.Tensor = torch.zeros(len(self.__vocab))
                else:
                    prediction_tokens_len: int = self.__source_len2answer_len.get(source_len, 10)
                    prediction_tokens = [
                        key for key, value in self.__counter.most_common(prediction_tokens_len)
                    ]

                    prediction_tokens_codes: list[int] = [self.__vocab[token] for token in prediction_tokens]

                    prediction_one_hot = torch.nn.functional.one_hot(
                        torch.tensor(prediction_tokens_codes).long(),
                        num_classes=len(self.__vocab),
                    )
                    prediction_one_hot = prediction_one_hot.sum(dim=0).float()

                if answer_str != "":
                    answer_unique_tokens = list(set(answer_str.split(" ")))

                    answer_tokens_codes: list[int] = [self.__vocab[token] for token in answer_unique_tokens]

                    answer_one_hot: torch.Tensor = torch.nn.functional.one_hot(
                        torch.tensor(answer_tokens_codes).long(),
                        num_classes=len(self.__vocab),
                    )
                    answer_one_hot = answer_one_hot.sum(dim=0).float()
                else:
                    answer_one_hot = torch.zeros(len(self.__vocab))

                for metric in metrics_storage:
                    metrics_storage[metric].append(
                        METRIC_REGISTER[metric](
                            answer_one_hot.numpy(),
                            prediction_one_hot.numpy(),
                        )
                    )

        if self.__use_confidence_interval and self.__interval_type not in ["standart", "quantile", "normal", "central"]:
            warnings.warn("'interval_type' you've passed doen't supported. See docs for more details. \n" + \
                            "Only point estimation will be returned.")
            point_estimations: dict[str, float] = self._get_point_estimation(metrics_storage)

            return point_estimations

        if not self.__use_confidence_interval:
            point_estimations = self._get_point_estimation(metrics_storage)

            return point_estimations

        confidence_intervals = self._get_confidence_interval(metrics_storage)

        return confidence_intervals
