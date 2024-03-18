import numpy as np
import torch
import typing as tp
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch.utils.data import DataLoader
from tqdm import tqdm


class FillMaskPipelineResult(tp.TypedDict):
    score: float
    token: int
    token_str: str


class MetricComputer:
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

    @staticmethod
    def __compute_reciprocal_rank(answer: str, predicted_tokens: list[str]) -> float:
        """
        Computes reciprocal rank over one example.

        Args:
            answer (str): true token's value;
            predicted_tokens (list[str]): tokens you model predicted.

        Returns:
            float: reciprocal rank's value.
        """
        if answer in predicted_tokens:
            return 1 / (predicted_tokens.index(answer) + 1)

        return 0.0

    @staticmethod
    def __compute_simplified_dcg(answer: str, predicted_tokens: list[str]) -> float:
        """
        Computes simpified DCG over one example.

        Args:
            answer (str): true token's value;
            predicted_tokens (list[str]): tokens you model predicted.

        Returns:
            float: simpified DCG's value.
        """
        if answer in predicted_tokens:
            return (1 / np.log2((predicted_tokens.index(answer) + 1) + 1)).astype(float)  # type: ignore 

        return 0.0

    @staticmethod
    def __compute_precision(answer: str, predicted_tokens: list[str]) -> float:
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

    @staticmethod
    def __compute_recall(answer: str, predicted_tokens: list[str]) -> float:
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

    @staticmethod
    def __compute_f_score(precision: float, recall: float) -> float:
        """
        Computes F_1 score from given precision and recall.

        Args:
            precision (float): precision's value on some example.
            recall (float): recall's value on the same example.

        Returns:
            float: F_1 score.
        """
        eps = 1e-9
        return (2 * precision * recall) / (precision + recall + eps)

    @staticmethod
    def __compute_hits(answer: str, predicted_tokens: list[str]) -> int:
        """
        Computes HITS value on one example;

        Args:
            answer (str): true token's value;
            predicted_tokens (list[str]): tokens you model predicted.

        Returns:
            int: HITS' value.
        """
        return predicted_tokens.count(answer)

    def get_metrics_value(
        self,
        model: PreTrainedModel,
        reciprocal_rank: bool = False,
        simplified_dcg: bool = False,
        precision: bool = False,
        recall: bool = False,
        f_score: bool = False,
        hit: bool = False,
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

        if reciprocal_rank:
            reciprocal_ranks: list[float] = []

        if simplified_dcg:
            simplified_dcgs: list[float] = []

        if precision:
            precisions: list[float] = []

        if recall:
            recalls: list[float] = []

        if f_score:
            f_scores: list[float] = []

        if hit:
            hits: list[float] = []

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

                    if reciprocal_rank:
                        reciprocal_ranks.append(
                            self.__compute_reciprocal_rank(answer, predicted_tokens)
                        )

                    if simplified_dcg:
                        simplified_dcgs.append(
                            self.__compute_simplified_dcg(answer, predicted_tokens)
                        )

                    if precision:
                        precisions.append(
                            self.__compute_precision(answer, predicted_tokens)
                        )

                    if recall:
                        recalls.append(self.__compute_recall(answer, predicted_tokens))

                    if f_score:
                        f_scores.append(
                            self.__compute_f_score(precisions[-1], recalls[-1])
                        )

                    if hit:
                        hits.append(self.__compute_hits(answer, predicted_tokens))

        metrics_value: dict[str, float] = {}
        if reciprocal_rank:
            metrics_value["reciprocal_rank"] = np.mean(reciprocal_ranks).astype(float)

        if simplified_dcg:
            metrics_value["simplified_dcg"] = np.mean(simplified_dcgs).astype(float)

        if precision:
            metrics_value["precision"] = np.mean(precisions).astype(float)

        if recall:
            metrics_value["recall"] = np.mean(recalls).astype(float)

        if f_score:
            metrics_value["f_score"] = np.mean(f_scores).astype(float)

        if hit:
            metrics_value["hit"] = np.mean(hits).astype(float)

        return metrics_value
