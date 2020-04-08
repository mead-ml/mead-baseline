"""A collection of multi-label metrics from here

https://www.researchgate.net/profile/Mohammad_Sorower/publication/266888594_A_Literature_Survey_on_Algorithms_for_Multi-label_Learning/links/58d1864392851cf4f8f4b72a/A-Literature-Survey-on-Algorithms-for-Multi-label-Learning.pdf
"""

from typing import Dict
import numpy as np
from eight_mile.utils import recall as em_recall, precision as em_precision


# Example oriented metrics
def exact_match(golds: np.ndarray, preds: np.ndarray) -> float:
    """Calculate the proportion of examples where the gold and predicted labels match exactly.

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns: The exact match score
    """
    return np.mean(np.all(golds == preds, axis=1))


def accuracy(golds: np.ndarray, preds: np.ndarray) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    return np.mean(np.sum(golds & preds, axis=1) / np.sum(golds | preds, axis=1))


def precision(golds: np.ndarray, preds: np.ndarray) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    correct = np.sum(golds & preds, axis=1)
    denom = np.sum(preds, axis=1)
    denom[denom == 0.0] = 1
    return np.mean(correct / denom)



def recall(golds: np.ndarray, preds: np.ndarray) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    return np.mean(np.sum(golds & preds, axis=1) / np.sum(golds, axis=1))


def f_score(golds: np.ndarray, preds: np.ndarray, beta: int = 1) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    beta_sq = beta ** 2
    correct = np.sum(golds & preds, axis=1)
    guessed = np.sum(preds, axis=1)
    guessed[guessed == 0] = 1
    p = correct / guessed
    real = np.sum(golds, axis=1)
    real[real == 0] = 1
    r = correct / real
    denom = beta_sq * p + r
    denom[denom == 0.0] = 1
    f = (1 + beta_sq) * p * r / denom
    return np.mean(f)


def hamming_loss(golds: np.ndarray, preds: np.ndarray) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    return np.count_nonzero(golds - preds) / np.prod(golds.shape)


# Class oriented metrics
def macro_precision(golds: np.ndarray, preds: np.ndarray) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    correct = np.sum(golds & preds, axis=0)
    guessed = np.sum(preds, axis=0)
    guessed[guessed == 0] = 1.0
    return np.mean(correct / guessed)


def macro_recall(golds: np.ndarray, preds: np.ndarray) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    correct = np.sum(golds & preds, axis=0)
    total = np.sum(golds, axis=0)
    total[total == 0] = 1.0
    return np.mean(correct / total)


def macro_f_score(golds: np.ndarray, preds: np.ndarray, beta: int = 1) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    beta_sq = beta ** 2
    p = macro_precision(golds, preds)
    r = macro_recall(golds, preds)
    if p == 0 or r == 0:
        return 0.0
    return (1 + beta_sq) * p * r / (beta_sq * p + r)


def micro_precision(golds: np.ndarray, preds: np.ndarray) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    correct = np.sum(golds & preds)
    guessed = np.sum(preds)
    return em_precision(correct, guessed)


def micro_recall(golds: np.ndarray, preds: np.ndarray) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    correct = np.sum(golds & preds)
    true = np.sum(golds)
    return em_recall(correct, true)


def micro_f_score(golds: np.ndarray, preds: np.ndarray, beta: int = 1) -> float:
    """

    :param golds: A [N, K] array representing the multihot gold labels for N
        examples with K possible classes. For a given example i and class j
        golds[i, j] == 1 if that example has a label of class j, == 0 otherwise
    :param preds: A [N, K] array representing the multihot labels for N examples
        with K possible classes produced by a classifer.

    :returns:
    """
    beta_sq = beta ** 2
    p = micro_precision(golds, preds)
    r = micro_recall(golds, preds)
    if p == 0 or r == 0:
        return 0.0
    return (1 + beta_sq) * p * r / (beta_sq * p + r)


def get_all_metrics(golds: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "exact_match": exact_match(golds, preds),
        "accuracy": accuracy(golds, preds),
        "precision": precision(golds, preds),
        "recall": recall(golds, preds),
        "f1": f_score(golds, preds, 1),
        "hamming_loss": hamming_loss(golds, preds),
        "macro_precision": macro_precision(golds, preds),
        "macro_recall": macro_recall(golds, preds),
        "macro_f1": macro_f_score(golds, preds, 1),
        "micro_precision": micro_precision(golds, preds),
        "micro_recall": micro_recall(golds, preds),
        "micro_f1": micro_f_score(golds, preds, 1),
    }
