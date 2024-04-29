import warnings
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import typing as tp
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    @abstractmethod
    def get_metrics_value(self, *args, **kwargs) -> dict[str, float]:  # type: ignore
        pass

    @metric
    @staticmethod
    def _compute_exact_match(answer: np.ndarray, predicted_tokens: np.ndarray) -> float:
        return float(np.all(answer == predicted_tokens))

    @metric
    @staticmethod
    def _compute_accuracy(answer: np.ndarray, predicted_tokens: np.ndarray) -> float:
        eps: float = 1e-9

        return float(sum(np.logical_and(answer, predicted_tokens)) / (sum(np.logical_or(answer, predicted_tokens)) + eps))

    @metric
    @staticmethod
    def _compute_recall(answer: np.ndarray, predicted_tokens: np.ndarray) -> float:
        if sum(answer) == 0:
            return 0.

        return float(sum(np.logical_and(answer, predicted_tokens)) / sum(answer))

    @metric
    @staticmethod
    def _compute_precision(answer: np.ndarray, predicted_tokens: np.ndarray) -> float:
        if sum(predicted_tokens) == 0:
            return 0.

        return float(sum(np.logical_and(answer, predicted_tokens)) / sum(predicted_tokens))

    @metric
    @staticmethod
    def _compute_f_score(answer: np.ndarray, predicted_tokens: np.ndarray, beta: float = 1.) -> float:
        eps: float = 1e-9

        precision: float = MetricComputerInterface._compute_precision(answer, predicted_tokens)
        recall: float = MetricComputerInterface._compute_recall(answer, predicted_tokens)

        return ((1 + beta**2) * precision * recall) / ((beta**2) * precision + recall + eps)


class MultiLabelHeadMetricComputer(MetricComputerInterface):
    def __init__(
        self,
        threshold: float,
        dataloader: DataLoader,
        device: torch.device,
    ):
        self.threshold: float = threshold
        self.dataloader: DataLoader = dataloader
        self.device: torch.device = device

    def get_metrics_value(
        self,
        model: torch.nn.Module,
        metrics_to_use: dict[str, bool],
    ) -> dict[str, float]:
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

        metrics_value: dict[str, float] = {}

        for metric in metrics_storage:
            metrics_value[metric] = float(np.mean(metrics_storage[metric]))

        return metrics_value
