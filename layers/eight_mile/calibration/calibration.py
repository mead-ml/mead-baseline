from typing import Optional
from collections import namedtuple
import numpy as np

__all__ = ["calibration_bins", "multiclass_calibration_bins", "binary_calibration_bins", "average_confidence", "Bins"]


Bins = namedtuple("Bins", "accs confs counts edges")


def multiclass_calibration_bins(truth: np.ndarray, probs: np.ndarray, bins: int, class_weights: Optional[np.ndarray] = None) -> Bins:
    """Calculate the binned confidence and accuracy for a multiclass problem.

    :param truth: A 1D array of the true labels for some examples.
    :param probs: A 1D array of the probabilities from a model. Each row represents an example in the dataset.
        and each column represents the probably assigned by the model to each class for that example.
    :param bins: The number of bins to use when aggregating.
    :param class_weights: A 1D array of scores that can add extra weight to examples of specific classes.

    :returns: The metrics aggregated by bins
    """
    preds = np.argmax(probs, axis=1)
    pred_probs = np.max(probs, axis=1)
    credit = truth == preds
    if class_weights is not None:
        weighted_labels = class_weights[truth]
        credit = credit * weighted_labels
    return binary_calibration_bins(credit, pred_probs, bins=bins)


calibration_bins = multiclass_calibration_bins


def binary_calibration_bins(credit: np.ndarray, probs: np.ndarray, bins: int) -> Bins:
    """Calculate the accuracy and confidence inside bins. This is similar to the sklearn function.

    Note:
        This is different than the multiclass function. This will look at the scores for a specific
        class even if it is below the 1/num_classes threshold that will cause the multiclass
        version to not select this class.

    :param credit: How much credit the model gets for this example. If it is wrong the value will be zero
        If it is right one. You can also give different classes different weights by passing real values.
    :param probs: The probabilities assigned to the positive class by the model.
    :param bins: The number of bins to use when aggregating.

    :returns: The metrics aggregated by bins
    """
    bins = np.linspace(0.0, 1.0, num=bins + 1, endpoint=True)

    bin_idx = np.digitize(probs, bins) - 1

    bin_conf_sum = np.bincount(bin_idx, weights=probs, minlength=len(bins))
    bin_acc_sum = np.bincount(bin_idx, weights=credit, minlength=len(bins))
    bin_counts = np.bincount(bin_idx, minlength=len(bins))

    mask = bin_counts == 0
    denom = bin_counts + mask

    bin_mean_conf = bin_conf_sum / denom
    bin_mean_acc = bin_acc_sum / denom

    return Bins(bin_mean_acc[:-1], bin_mean_conf[:-1], bin_counts[:-1], bins[:-1])


def average_confidence(probs: np.ndarray) -> float:
    """Calculate the average (maximum) confidence for a collection of predictions

    :param probs: `[B, C]` A matrix of probabilities, each row is an example and
        each column is a class.
    """
    if probs.ndim == 1:
        probs = np.expand_dims(probs, axis=-1)
    return np.mean(np.max(probs, axis=1))
