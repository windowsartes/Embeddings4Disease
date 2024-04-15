import typing as tp
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, Counter

import numpy as np
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch.utils.data import DataLoader
from tqdm import tqdm


class FillMaskPipelineResult(tp.TypedDict):
    score: float
    token: int
    token_str: str


METRIC_REGISTER = {}

def metric(function: tp.Callable[[str, list[str], list[float]], float]) -> tp.Callable[[str, list[str], list[float]], float]:
    METRIC_REGISTER[function.__name__[10:]] = function
    return function


class MetricComputerInterface(ABC):
    @abstractmethod
    def get_metrics_value(self, *args, **kwargs) -> dict[str, float]: # type: ignore
        pass

    @metric
    @staticmethod
    def __compute_reciprocal_rank(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        Computes reciprocal rank over one example.

        Args:
            answer (str): true token's value;
            predicted_tokens (list[str]): tokens you model predicted.

        Returns:
            float: reciprocal rank's value.
        """
        if answer in predicted_tokens:
            return 1. / (predicted_tokens.index(answer) + 1)

        return 0.0

    @metric
    @staticmethod
    def __compute_simplified_dcg(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        Computes simpified DCG over one example.

        Args:
            answer (str): true token's value;
            predicted_tokens (list[str]): tokens you model predicted.

        Returns:
            float: simpified DCG's value.
        """
        if answer in predicted_tokens:
            return (1. / np.log2((predicted_tokens.index(answer) + 1) + 1)).astype(float)  # type: ignore

        return 0.0

    @metric
    @staticmethod
    def __compute_precision(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        Computes precision over one example.

        Args:
        answer (str): true token's value;
        predicted_tokens (list[str]): tokens you model predicted.


        Returns:
            float: precision's score.
        """
        TP: int = predicted_tokens.count(answer)
        TP_and_FP: int = len(predicted_tokens)

        return TP / TP_and_FP

    @metric
    @staticmethod
    def __compute_recall(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        """
        Computes recall over one example.

        Note that:
        TP means we've predicted the answer; there can be only 1 answer
        FN means we haven't predicted the answer; since there can obly be 1 answer,
        it's 1 when the answer doesn't belong to predicted_tokens,
        so it's always equals to 1

        Args:
            answer (str): true token's value;
            predicted_tokens (list[str]): tokens you model predicted._

        Returns:
            float: recall's value.
        """

        TP: int = predicted_tokens.count(answer)
        TP_and_FN: int = 1

        return TP / TP_and_FN

    @metric
    @staticmethod
    def __compute_f_score(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float], beta: float = 1.) -> float:
        """
        TBA

        Args:
            answer (str): _description_
            predicted_tokens (list[str]): _description_
            beta (falot, optional): _description_. Defaults to 1..

        Returns:
            float: _description_
        """
        eps = 1e-9

        precision = MLMMetricComputer.__compute_precision(answer, predicted_tokens, tokens_probabilities)
        recall = MLMMetricComputer.__compute_recall(answer, predicted_tokens, tokens_probabilities)

        return ((1 + beta**2) * precision * recall) / ((beta**2) * precision + recall + eps)

    @metric
    @staticmethod
    def __compute_confidence(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> float:
        for token, probability in zip(predicted_tokens, tokens_probabilities):
            if token == answer:
                return probability

        return 0.

    @metric
    @staticmethod
    def __compute_hits(answer: str, predicted_tokens: list[str], tokens_probabilities: list[float]) -> int:
        """
        Computes HITS value on one example;

        Args:
            answer (str): true token's value;
            predicted_tokens (list[str]): tokens you model predicted.

        Returns:
            int: HITS' value.
        """
        return predicted_tokens.count(answer)


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
        tokenizer: PreTrainedTokenizer,
        top_k: int,
        dataloader: DataLoader,
    ):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.top_k: int = top_k
        self.dataloader: DataLoader = dataloader

    def get_metrics_value(
        self,
        model: PreTrainedModel,
        metrics_to_use: dict[str, bool],
    ) -> dict[str, float]:
        """
        Using this method you can evaluate your model. It will compute metric's value on all the data from
        the dataloader and return it's average score. For now it supports only these metrics:
        reciprocal rank, simplified DCG, precision, recall, F_1 score and HITS. In the case you want add an
        extra one, follow this guide:
        1. add metric's name as a boolean method parameter;
        2. add private method for that computes your metric's value over 1 object;
        3. don't forget to initialize an empty list for metric's values, call private function during evaluation
            after fill-mask pipeline, compute and return metric's average score as a dict key-value pair;
        4. Also you want to use this metric during cli training or inference, add it to the config file;

        Args:
            model (PreTrainedModel): Bert-like model which performance you want to evaluate.
            reciprocal_rank (bool, optional): compute or not reciprocal rank . Defaults to False.
            simplified_dcg (bool, optional): compute or not simplified DCG. Defaults to False.
            precision (bool, optional): compute or not precision. Defaults to False.
            recall (bool, optional): compute or not recall. Defaults to False.
            f_score (bool, optional): compute or not F_1 score. By design it's True only
                        if recall and precision are both True. Defaults to False.
            hit (bool, optional): compute or not HITS. Defaults to False.

        Returns:
            dict[str, float]: a dictionary with metric's average metrics values over dataloader's data
        """
        metrics_storage: dict[str, list[float]] = {}
        for metric, usage in metrics_to_use.items():
            if usage:
                if metric not in METRIC_REGISTER:
                    warnings.warn(
                        f"There is no {metric} in the supported metrics so this key will be ignored",
                        SyntaxWarning
                    )
                else:
                    metrics_storage[metric] = []

        with torch.no_grad():
            for batch_inputs, batch_answers in tqdm(self.dataloader):
                outputs = model(**batch_inputs)

                results: list[list[FillMaskPipelineResult]] = []

                for i in range(outputs.logits.shape[0]):
                    masked_index = torch.nonzero(
                        batch_inputs["input_ids"][i] == self.tokenizer.mask_token_id,
                        as_tuple=False,
                    )

                    logits = outputs.logits[i, masked_index[0].item(), :]
                    probabilities = logits.softmax(dim=0)

                    top_probabilities, top_predicted_tokens = probabilities.topk(
                        self.top_k
                    )

                    result: list[FillMaskPipelineResult] = []

                    for probability, predicted_token in zip(
                        top_probabilities.tolist(), top_predicted_tokens.tolist()
                    ):
                        result.append(
                            {
                                "score": probability,
                                "token": predicted_token,
                                "token_str": self.tokenizer.decode([predicted_token]),
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

        metrics_value: dict[str, float] = {}

        for metric in metrics_storage:
            metrics_value[metric] = float(np.mean(metrics_storage[metric]))

        return metrics_value


class Baseline(MetricComputerInterface):
    def __init__(self,
                 config: dict[str, tp.Any],
        ):
        super().__init__()

        counter: tp.DefaultDict[str, int] = defaultdict(int)

        with open(config["path_to_training_file"], "r") as training_file:
            for transaction in training_file:
                tokens = transaction.strip().split(" ")
                for token in tokens:
                    counter[token] += 1

        number_of_tokens: int = sum(counter.values())
        self._config = config
        self._counter: Counter[str] = Counter(dict(sorted(counter.items(), key=lambda item: item[1], reverse=True)))
        self._top_k_predictions: dict[str, float] = {key: value/number_of_tokens for key, value in self._counter.most_common(config["top_k"])}
        # true_counter.most_common(top_k)

    def get_metrics_value(
        self,
    ) -> dict[str, float]:
        """

        Args:
            path_to_validation_file (str | pathlib.Path): _description_
            metrics_to_use (dict[str, float]): _description_

        Returns:
            dict[str, float]: _description_
        """
        metrics_storage: dict[str, list[float]] = {}
        for metric, usage in self._config["metrics"].items():
            if usage:
                if metric not in METRIC_REGISTER:
                    warnings.warn(
                        f"There is no {metric} in the supported metrics so this key will be ignored",
                        SyntaxWarning
                    )
                else:
                    metrics_storage[metric] = []

        predicted_tokens: list[str] = list(self._top_k_predictions.keys())
        tokens_probabilities: list[float] = list(self._top_k_predictions.values())

        with open(self._config["path_to_validation_file"], "r") as validation_file:
            for input_sequence in validation_file:
                answer: str = input_sequence.strip().split(" ")[-1]

                for metric in metrics_storage:
                    metrics_storage[metric].append(METRIC_REGISTER[metric](answer, predicted_tokens, tokens_probabilities))

        metrics_value: dict[str, float] = {}

        for metric in metrics_storage:
            metrics_value[metric] = float(np.mean(metrics_storage[metric]))

        return metrics_value
