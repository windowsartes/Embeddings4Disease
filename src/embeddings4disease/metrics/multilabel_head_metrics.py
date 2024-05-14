import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.stats as ss  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
import transformers
import typing as tp
from torch.utils.data import DataLoader
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
        self._confidence_interval: bool = False
        self._interval_type: tp.Optional[str] = None
        self._confidence_level: float = 0.95

    @abstractmethod
    def get_metrics_value(self, *args, **kwargs) -> dict[str, float] | dict[str, ConfidenceInterval]:  # type: ignore
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
        if sum(answer) == 0:
            return 0.

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
        if sum(predicted_tokens) == 0:
            return 0.

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

            if self._interval_type == "normal":
                bootstrap_estimations_std = np.std(bootstrap_estimations, ddof=1)
                quantile = ss.norm.ppf((1 + self._confidence_level) / 2, loc=0, scale=1)

                metrics_value[metric] = ConfidenceInterval(
                    float(round(point_estimation, 4)),
                    (
                        float(round(point_estimation - quantile * bootstrap_estimations_std, 4)),
                        float(round(point_estimation + quantile * bootstrap_estimations_std, 4))
                    ),
                )

            elif self._interval_type == "quantile":
                bootstrap_estimations_sorted = sorted(bootstrap_estimations)

                metrics_value[metric] = ConfidenceInterval(
                    float(round(point_estimation, 4)),
                    (
                        float(round(bootstrap_estimations_sorted[math.floor(n_resamples * \
                            ((1 - self._confidence_level) / 2))], 4)),
                        float(round(bootstrap_estimations_sorted[math.ceil(n_resamples * \
                            ((1 + self._confidence_level) / 2))], 4)),
                    ),
                )
            else:
                bootstrap_estimations_sorted = sorted(bootstrap_estimations)

                metrics_value[metric] = ConfidenceInterval(
                    float(round(point_estimation, 4)),
                    (
                        float(round(2 * point_estimation - \
                            bootstrap_estimations_sorted[math.ceil(n_resamples * \
                                ((1 + self._confidence_level) / 2))], 4)),
                        float(round(2 * point_estimation - \
                            bootstrap_estimations_sorted[math.floor(n_resamples * \
                                ((1 - self._confidence_level) / 2))], 4)),
                    ),
                )

        return metrics_value


class MultiLabelHeadMetricComputer(MetricComputerInterface):
    def __init__(
        self,
        threshold: float,
        dataloader: DataLoader,
        device: torch.device,
        confidence_interval: bool = False,
        interval_type: tp.Optional[str] = None,
        confidence_level: float = 0.95,
    ):
        self.threshold: float = threshold
        self.dataloader: DataLoader = dataloader
        self.device: torch.device = device

        self._confidence_interval: bool = confidence_interval
        self._interval_type: tp.Optional[str] = interval_type
        self._confidence_level: float = confidence_level

    def get_metrics_value(
        self,
        model: torch.nn.Module,
        metrics_to_use: dict[str, bool],
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
        for metric, usage in metrics_to_use.items():
            if usage:
                if metric not in METRIC_REGISTER:
                    warnings.warn(
                        f"There is no {metric} in the supported metrics so it will be ignored",
                        SyntaxWarning
                    )
                else:
                    metrics_storage[metric] = []

        with torch.no_grad():
            progress_bar = tqdm(self.dataloader)
            for batch_inputs, batch_answers in progress_bar:
                progress_bar.set_description("Computing metrics")

                batch_inputs = batch_inputs.to(self.device)
                outputs = model(batch_inputs)

                probabilities: torch.Tensor = F.sigmoid(outputs)

                predictions: npt.NDArray[np.float64] = (probabilities > self.threshold).float().detach().cpu().numpy()

                for answer, predicted_tokens in zip(batch_answers, predictions):
                    for metric in metrics_storage:
                        metrics_storage[metric].append(METRIC_REGISTER[metric](answer.numpy(), predicted_tokens))
            progress_bar.close()

        if self._confidence_interval and self._interval_type not in ["quantile", "normal", "central"]:
            warnings.warn("'interval_type' you've passed doen't supported. See docs for more details. \n" + \
                          "Only point estimation will be returned.")
            return self._get_point_estimation(metrics_storage)

        if not self._confidence_interval:
            return self._get_point_estimation(metrics_storage)

        return self._get_confidence_interval(metrics_storage)


class EncoderDecoderHeadMetricComputer(MetricComputerInterface):
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        max_length: int,
        confidence_interval: bool = False,
        interval_type: tp.Optional[str] = None,
        confidence_level: float = 0.95,
    ):
        self.dataloader: DataLoader = dataloader
        self.device: torch.device = device

        self.tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast = tokenizer
        self.vocab: dict[str, str] = tokenizer.get_vocab()

        self.max_length: int = max_length

        self._confidence_interval: bool = confidence_interval
        self._interval_type: tp.Optional[str] = interval_type
        self._confidence_level: float = confidence_level

    def get_metrics_value(
        self,
        model: torch.nn.Module,
        metrics_to_use: dict[str, bool],
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
        for metric, usage in metrics_to_use.items():
            if usage:
                if metric not in METRIC_REGISTER:
                    warnings.warn(
                        f"There is no {metric} in the supported metrics so it will be ignored",
                        SyntaxWarning
                    )
                else:
                    metrics_storage[metric] = []

        with torch.no_grad():
            progress_bar = tqdm(self.dataloader)
            for source_strings, target_strings in progress_bar:
                progress_bar.set_description("Computing metrics")

                device = next(model.parameters()).device

                encoded_input = self.tokenizer(
                    source_strings,
                    padding='longest',
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                decoder_input = self.tokenizer(
                    source_strings,
                    padding='longest',
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                answer_batch = self.tokenizer(
                    target_strings,
                    return_tensors="np",
                ).input_ids

                model_predictions = model.generate(
                    input_ids=encoded_input["input_ids"],
                    decoder_input_ids=decoder_input["input_ids"],
                    max_length=self.max_length + 1,
                )

                for answer, predicted_tokens in zip(answer_batch, model_predictions.cpu()):
                    for _, special_token in self.tokenizer.special_tokens_map.items():
                        predicted_tokens = predicted_tokens[predicted_tokens != self.vocab[special_token]]
                        answer = answer[answer != self.vocab[special_token]]

                    answer_one_hot: torch.Tensor = torch.nn.functional.one_hot(
                        torch.tensor(answer).long(), num_classes=len(self.vocab),
                    ).sum(dim=0).float()

                    predictions_one_hot: torch.Tensor = torch.nn.functional.one_hot(
                        predicted_tokens.long(),
                        num_classes=len(self.vocab),
                    ).sum(dim=0).float()

                    for metric in metrics_storage:
                        metrics_storage[metric].append(METRIC_REGISTER[metric](
                            answer_one_hot.numpy(),
                            predictions_one_hot.numpy()
                        ))
            progress_bar.close()

        if self._confidence_interval and self._interval_type not in ["quantile", "normal", "central"]:
            warnings.warn("'interval_type' you've passed doen't supported. See docs for more details. \n" + \
                          "Only point estimation will be returned.")
            return self._get_point_estimation(metrics_storage)

        if not self._confidence_interval:
            return self._get_point_estimation(metrics_storage)

        return self._get_confidence_interval(metrics_storage)
