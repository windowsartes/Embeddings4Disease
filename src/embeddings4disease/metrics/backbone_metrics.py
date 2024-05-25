import math
import typing as tp
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np
import scipy.stats as ss  # type: ignore[import-untyped]
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput
from torch.utils.data import DataLoader
from tqdm import tqdm


class FillMaskPipelineResult(tp.TypedDict):
    score: float
    token: int
    token_str: str


@dataclass
class ConfidenceInterval:
    estimation: float
    interval: tuple[float, float]


METRIC_REGISTER: dict[str, tp.Callable[[str, list[str], list[float]], float]] = {}

def metric(function: tp.Callable[[str, list[str], list[float]], float]) -> tp.Callable[[str, list[str], list[float]], float]:
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

    @abstractmethod
    def get_metrics_value(self, *args, **kwargs) -> dict[str, float] | dict[str, ConfidenceInterval]: # type: ignore
        pass

    @metric
    @staticmethod
    def _compute_reciprocal_rank(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        Computes reciprocal rank over one example.

        Args:
            answer (str): true token's value.
            predicted_tokens (list[str]): tokens you model has predicted.
            tokens_probabilities (list[float]): probabilities of corresponding tokens.

        Returns:
            float: reciprocal rank's value.
        """
        if answer in predicted_tokens:
            return 1. / (predicted_tokens.index(answer) + 1)

        return 0.0

    @metric
    @staticmethod
    def _compute_simplified_dcg(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        Computes simpified DCG over one example.

        Args:
            answer (str): true token's value.
            predicted_tokens (list[str]): tokens you model has predicted.
            tokens_probabilities (list[float]): probabilities of corresponding tokens.

        Returns:
            float: simpified DCG's value.
        """
        if answer in predicted_tokens:
            return (1. / np.log2((predicted_tokens.index(answer) + 1) + 1)).astype(float)  # type: ignore

        return 0.0

    @metric
    @staticmethod
    def _compute_precision(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        Computes precision over one example.

        Args:
            answer (str): true token's value.
            predicted_tokens (list[str]): tokens you model has predicted.
            tokens_probabilities (list[float]): probabilities of corresponding tokens.

        Returns:
            float: precision's score.
        """
        TP: int = predicted_tokens.count(answer)
        TP_and_FP: int = len(predicted_tokens)

        return TP / TP_and_FP

    @metric
    @staticmethod
    def _compute_recall(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        Computes recall over one example.

        Note that:
        TP means we've predicted the answer; there can be only 1 answer
        FN means we haven't predicted the answer; since there can obly be 1 answer,
        it's 1 when the answer doesn't belong to predicted_tokens,
        so it's always equals to 1

        Args:
            answer (str): true token's value.
            predicted_tokens (list[str]): tokens you model has predicted.
            tokens_probabilities (list[float]): probabilities of corresponding tokens.

        Returns:
            float: recall's value.
        """

        TP: int = predicted_tokens.count(answer)
        TP_and_FN: int = 1

        return TP / TP_and_FN

    @metric
    @staticmethod
    def _compute_f_score(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float], beta: float = 1.) -> float:
        """
        Computes f-beta score.

        Args:
            answer (str): true token's value.
            predicted_tokens (list[str]): tokens you model has predicted.
            tokens_probabilities (list[float]): probabilities of corresponding tokens.
            beta (falot, optional): beta coefficient. Defaults to 1.

        Returns:
            float: f-score's value.
        """
        eps: float = 1e-9

        precision: float = MetricComputerInterface._compute_precision(answer, predicted_tokens, tokens_probabilities)
        recall: float = MetricComputerInterface._compute_recall(answer, predicted_tokens, tokens_probabilities)

        return ((1 + beta**2) * precision * recall) / ((beta**2) * precision + recall + eps)

    @metric
    @staticmethod
    def _compute_confidence(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        There confidence is a probability of a true token in the case you model has predicted it, else 0.

        Args:
            answer (str): true token's value.
            predicted_tokens (list[str]): tokens you model has predicted.
            tokens_probabilities (list[float]): probabilities of corresponding tokens.

        Returns:
            float: confidence score.
        """
        for token, probability in zip(predicted_tokens, tokens_probabilities):
            if token == answer:
                return probability

        return 0.

    @metric
    @staticmethod
    def _compute_hits(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        Computes HITS value on one example;

        Args:
            answer (str): true token's value.
            predicted_tokens (list[str]): tokens you model has predicted.
            tokens_probabilities (list[float]): probabilities of corresponding tokens.

        Returns:
            float: HITS' value.
        """
        return predicted_tokens.count(answer)

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

            if self.__interval_type == "with_std":
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


class MLMMetricComputer(MetricComputerInterface):
    """
    Basic class for evalution model's performance.
    In the initializer you need to pass following argimnets:
    Args:
        tokenizer (PreTrainedTokenizer): tokenizer you want to use.
        top_k (int): top k predictions that model will prodice to replace [MASK]ed toke.
        dataloader (DataLoader): DataLoader which will produce the data during inference.

    Metrics you want to track and the model you want to evaluate you will specify in the
    'get_metrics_value' method.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        top_k: int,
        dataloader: DataLoader,
        metrics_to_use: dict[str, bool],
        confidence_interval: bool = False,
        interval_type: tp.Optional[str] = None,
        confidence_level: float = 0.95,
    ):
        super().__init__()

        self.__tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer
        self.__top_k: int = top_k
        self.__dataloader: DataLoader = dataloader

        self.__metrics_to_use: dict[str, bool] = metrics_to_use

        self.__use_confidence_interval: bool = confidence_interval
        self.__interval_type: tp.Optional[str] = interval_type
        self.__confidence_level: float = confidence_level

    def get_metrics_value(
        self,
        model: PreTrainedModel,
    ) -> dict[str, float] | dict[str, ConfidenceInterval]:
        """
        Base method for the model evaluation. It will computer metrics' values during the dataloader you've specified
        at the instance initialization.

        Args:
            model (PreTrainedModel): model you want to evaluate.
            metrics_to_use (dict[str, bool]): metrics you want to use during inference. In the case some metric you've
            passed in this dictionary doesn't supported, it'll be ignored.

        Returns:
            dict[str, float]: average metrics's values during evaluation.
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
            progress_bar.set_description("Computing metrics")

            for batch_inputs, batch_answers in progress_bar:
                outputs = model(**batch_inputs)

                results: list[list[FillMaskPipelineResult]] = []

                if type(outputs) == BertForPreTrainingOutput:
                    shape: int = outputs.prediction_logits.shape[0]
                else:
                    shape = outputs.logits.shape[0]

                for i in range(shape):
                    masked_index = torch.nonzero(
                        batch_inputs["input_ids"][i] == self.__tokenizer.mask_token_id,
                        as_tuple=False,
                    )

                    if type(outputs) == BertForPreTrainingOutput:
                        logits = outputs.prediction_logits[i, masked_index[0].item(), :]
                    else:
                        logits = outputs.logits[i, masked_index[0].item(), :]
                    probabilities = logits.softmax(dim=0)

                    top_probabilities, top_predicted_tokens = probabilities.topk(self.__top_k)

                    result: list[FillMaskPipelineResult] = []

                    for probability, predicted_token in zip(
                        top_probabilities.tolist(), top_predicted_tokens.tolist()
                    ):
                        result.append(
                            {
                                "score": probability,
                                "token": predicted_token,
                                "token_str": self.__tokenizer.decode([predicted_token]),
                            }
                        )

                    results += [result]

                for answer, model_predictions in zip(batch_answers, results):
                    predicted_tokens = [
                        prediction["token_str"] for prediction in model_predictions
                    ]
                    tokens_probabilities = [
                        prediction["score"] for prediction in model_predictions
                    ]

                    for metric in metrics_storage:
                        metrics_storage[metric].append(METRIC_REGISTER[metric](answer, predicted_tokens, tokens_probabilities))

        if self.__use_confidence_interval and self.__interval_type not in ["with_std", "quantile", "normal", "central"]:
            warnings.warn("'interval_type' you've passed doen't supported. See docs for more details. \n" + \
                          "Only point estimation will be returned.")
            return self._get_point_estimation(metrics_storage)

        if not self.__use_confidence_interval:
            return self._get_point_estimation(metrics_storage)

        return self._get_confidence_interval(metrics_storage)


class Baseline(MetricComputerInterface):
    """
    A simple baseline model. It just uses the most populat top_k tokens as a prediction.
    """
    def __init__(self,
                 config: dict[str, tp.Any],
        ):
        super().__init__()

        counter_dct: tp.DefaultDict[str, int] = defaultdict(int)

        with open(config["path_to_training_file"], "r") as training_file:
            for transaction in training_file:
                tokens = transaction.strip().split(" ")
                for token in tokens:
                    counter_dct[token] += 1

        number_of_tokens: int = sum(counter_dct.values())
        self.__config: dict[str, tp.Any] = config

        self.__use_confidence_interval: bool = self.__config["confidence_interval"]["use"]
        self.__interval_type: tp.Optional[str] = self.__config["confidence_interval"]["interval_type"]
        self.__confidence_level: float = self.__config["confidence_interval"]["confidence_level"]

        counter: Counter[str] = Counter(dict(sorted(counter_dct.items(), key=lambda item: item[1], reverse=True)))
        self.__top_k_predictions: dict[str, float] = {
            key: value/number_of_tokens for key, value in counter.most_common(config["top_k"])
        }

    def get_metrics_value(
        self,
    ) -> dict[str, float] | dict[str, ConfidenceInterval]:
        """
        Base method for the baseline evaluation. It will computer metrics values during the validation data you've specified
        at the config.

        Returns:
            dict[str, float]: average metrics's values during evaluation.
        """

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

        predicted_tokens: list[str] = list(self.__top_k_predictions.keys())
        tokens_probabilities: list[float] = list(self.__top_k_predictions.values())

        with open(self.__config["path_to_validation_file"], "r") as validation_file:
            progress_bar = tqdm(validation_file.readlines())
            progress_bar.set_description("Computing metrics")

            for input_sequence in progress_bar:
                answer: str = input_sequence.strip().split(" ")[-1]

                for metric in metrics_storage:
                    metrics_storage[metric].append(
                        METRIC_REGISTER[metric](
                            answer,
                            predicted_tokens,
                            tokens_probabilities,
                        )
                    )

        if self.__use_confidence_interval and self.__interval_type not in ["standart", "quantile", "normal", "central"]:
            warnings.warn("'interval_type' you've passed doen't supported. See docs for more details. \n" + \
                          "Only point estimation will be returned.")
            return self._get_point_estimation(metrics_storage)

        if not self.__use_confidence_interval:
            return self._get_point_estimation(metrics_storage)

        return self._get_confidence_interval(metrics_storage)
