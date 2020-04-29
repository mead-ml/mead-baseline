import numpy as np


__all__ = ["expected_calibration_error", "maximum_calibration_error"]


def expected_calibration_error(
    bin_mean_acc: np.ndarray,
    bin_mean_conf: np.ndarray,
    bin_counts: np.ndarray,
) -> float:
    """The difference in expectation between the confidence and the accuracy.

    This is an approximation of eq (2) as described in eq (5)
        from https://arxiv.org/abs/1706.04599

    This is the absolute difference between the confidence and accuracy of each
    bin weighted by the number of samples in each bin.

    :param bin_mean_acc: `[B]` an array with the accuracy of each bin as elements
    :param bin_meanc_conf: `[B]` an array with the mean confidence of each bin as elements
    :param bin_counts: `[B]` an array with the number of samples in each bin

    :returns: The ECE between the accuracy and confidence distributions via binning.
    """
    abs_differences = np.abs(bin_mean_acc - bin_mean_conf)
    coeff = bin_counts / np.sum(bin_counts)
    return np.sum(coeff * abs_differences)


def maximum_calibration_error(
    bin_mean_acc: np.ndarray,
    bin_mean_conf: np.ndarray,
    bin_counts: np.ndarray,
) -> float:
    """The worst case deviation between confidence and accuracy.

    This is an approximation of eq (4) as described in eq (5)
        from https://arxiv.org/abs/1706.04599

    This is the maximum absolute difference between the confidence and accuracy
    of each bin. The bin_counts are used to filter out bins that have no samples
    in them. This seems like it is very dependent on the number of bins you have
    and should therefore be kept constant when comparing models.

    :param bin_mean_acc:
    :param bin_mean_conf:
    :param bin_counts:

    :returns: The MCE between the accuracy and confidence distributions via binning.
    """
    abs_differences = np.abs(bin_mean_acc - bin_mean_conf)
    filtered = abs_differences[bin_counts != 0]
    return np.max(filtered)
