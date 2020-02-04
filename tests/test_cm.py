import math
from itertools import chain
import pytest
import numpy as np
from eight_mile.confusion import ConfusionMatrix


Y_TRUE = [2, 0, 2, 2, 0, 1, 3, 1, 3, 3, 3, 3, 4]
Y_PRED = [0, 0, 2, 2, 0, 2, 3, 3, 3, 1, 3, 2, 4]
LABELS = ["0", "1", "2", "3", "4"]


CLASS_PREC = [0.666667, 0.0, 0.5, 0.75, 1.0]
CLASS_RECALL = [1.0, 0.0, 0.666667, 0.6, 1.0]
CLASS_F1 = [0.8, 0.0, 0.571429, 0.666667, 1.0]
CLASS_SUPPORT = [2, 2, 3, 5, 1]
CLASS_MCC = 0.5
TOL = 1e-6


def make_mc_cm():
    cm = ConfusionMatrix(LABELS)
    for y_t, y_p in zip(Y_TRUE, Y_PRED):
        cm.add(y_t, y_p)
    return cm


def test_create_cm():
    gold = make_mc_cm()
    cm = ConfusionMatrix.create(Y_TRUE, Y_PRED)
    np.testing.assert_equal(gold._cm, cm._cm)


def test_mc_support():
    cm = make_mc_cm()
    support = cm.get_support()
    np.testing.assert_allclose(support, CLASS_SUPPORT, TOL)


def test_mc_precision():
    cm = make_mc_cm()
    prec = cm.get_precision()
    np.testing.assert_allclose(prec, CLASS_PREC, TOL)
    wp = cm.get_weighted_precision()
    np.testing.assert_allclose(wp, 0.5833333, TOL)
    mp = cm.get_mean_precision()
    np.testing.assert_allclose(mp, 0.5833333, TOL)


def test_mc_recall():
    cm = make_mc_cm()
    recall = cm.get_recall()
    np.testing.assert_allclose(recall, CLASS_RECALL, TOL)
    wr = cm.get_weighted_recall()
    np.testing.assert_allclose(wr, 0.6153846, TOL)
    mr = cm.get_mean_recall()
    np.testing.assert_allclose(mr, 0.65333333, TOL)


def test_mc_f1():
    cm = make_mc_cm()
    f1 = cm.get_class_f()
    np.testing.assert_allclose(f1, CLASS_F1, TOL)
    wf1 = cm.get_weighted_f()
    np.testing.assert_allclose(wf1, 0.5882784, TOL)


def test_mcc_example():
    Y_TRUE = [1, 1, 1, 0]
    Y_PRED = [1, 0, 1, 1]
    MCC_GOLD = -0.3333333333333333
    cm = ConfusionMatrix.create(Y_TRUE, Y_PRED)
    np.testing.assert_allclose(cm.get_mcc(), MCC_GOLD, TOL)


def test_binary_mcc():
    Y_TRUE = [1, 1, 1, 0]
    Y_PRED = [1, 0, 1, 1]
    MCC_GOLD = -0.3333333333333333
    cm = ConfusionMatrix([0, 1])
    cm.add_batch(Y_TRUE, Y_PRED)
    np.testing.assert_allclose(cm.get_mcc(), MCC_GOLD, TOL)


def test_mc_mcc():
    cm = make_mc_cm()
    np.testing.assert_allclose(cm.get_r_k(), CLASS_MCC)


def explicit_mcc(golds, preds):
    """This is the way to calculate MCC directly from a confusion matrix as described here:
        "https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    for g, p in zip(golds, preds):
        if g == p and g == 0:
            TP += 1
        elif g == p and p == 1:
            TN += 1
        elif g != p and p == 0:
            FP += 1
        elif g != p and p == 1:
            FN += 1
    num = TP * TN - FP * FN
    denom = float(math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    denom = denom if denom != 0.0 else 1.0
    return num / denom


def author_mcc(cm):
    """Calculate Matthews Correlation Coefficient as described in this original paper
        https://www.ncbi.nlm.nih.gov/pubmed/1180967
    """
    N = np.sum(cm, dtype=np.float32)
    S = np.sum(cm, axis=0, dtype=np.float64)[0] / N
    P = np.sum(cm, axis=1, dtype=np.float64)[0] / N
    TP = cm[0, 0]
    num = TP / N - S * P
    denom = np.sqrt(P * S * (1 - P) * (1 - S))
    denom = denom if denom != 0.0 else 1.0
    return num / denom


def test_mcc():
    def test():
        golds = np.concatenate((np.arange(2), np.random.randint(0, 2, size=np.random.randint(4, 100))), axis=0)
        preds = np.random.randint(0, 2, size=len(golds))
        cm = ConfusionMatrix.create(golds, preds)
        np.testing.assert_allclose(cm.get_mcc(), explicit_mcc(golds, preds), TOL)
        np.testing.assert_allclose(cm.get_mcc(), author_mcc(cm._cm), TOL, TOL)

    for _ in range(100):
        test()


def test_mcc_perfect():
    golds = np.concatenate((np.arange(2), np.random.randint(0, 2, size=np.random.randint(4, 100))), axis=0)
    preds = np.copy(golds)
    cm = ConfusionMatrix.create(golds, preds)
    np.testing.assert_allclose(cm.get_mcc(), 1.0, TOL)


def test_mcc_inverse():
    golds = np.concatenate((np.arange(2), np.random.randint(0, 2, size=np.random.randint(4, 100))), axis=0)
    preds = 1 - golds
    cm = ConfusionMatrix.create(golds, preds)
    np.testing.assert_allclose(cm.get_mcc(), -1.0, TOL)


def explicit_r_k(cm):
    """This is the way to calculate MCC directly from a confusion matrix as described here:
        "https://en.wikipedia.org/wiki/Matthews_correlation_coefficient#Multiclass_case
    """
    num = 0.0
    for k in range(len(cm)):
        for l in range(len(cm)):
            for m in range(len(cm)):
                num += cm[k, k] * cm[l, m] - cm[k, l] * cm[m, k]
    denom1 = 0
    for k in range(len(cm)):
        k_l = 0
        for l in range(len(cm)):
            k_l += cm[k, l]
        not_k_l = 0
        for k2 in range(len(cm)):
            if k == k2:
                continue
            for l2 in range(len(cm)):
                not_k_l += cm[k2, l2]
        denom1 += k_l * not_k_l
    denom2 = 0
    for k in range(len(cm)):
        k_l = 0
        for l in range(len(cm)):
            k_l += cm[l, k]
        not_k_l = 0
        for k2 in range(len(cm)):
            if k == k2:
                continue
            for l2 in range(len(cm)):
                not_k_l += cm[l2, k2]
        denom2 += k_l * not_k_l
    denom = np.sqrt(denom1) * np.sqrt(denom2)
    denom = denom if denom != 0.0 else 1
    return num / denom


def test_r_k():
    def test():
        C = np.random.randint(3, 11)
        golds = np.concatenate((np.arange(C), np.random.randint(0, C, size=np.random.randint(4, 100))), axis=0)
        preds = np.random.randint(0, C, size=len(golds))
        cm = ConfusionMatrix.create(golds, preds)
        np.testing.assert_allclose(cm.get_r_k(), explicit_r_k(cm._cm), TOL)

    for _ in range(100):
        test()


def test_r_k_perfect():
    """Perfect correlation results in a score of 1."""
    C = np.random.randint(2, 11)
    golds = np.concatenate((np.arange(C), np.random.randint(0, C, size=np.random.randint(4, 100))), axis=0)
    preds = np.copy(golds)
    cm = ConfusionMatrix.create(golds, preds)
    np.testing.assert_allclose(cm.get_r_k(), 1.0)


def test_r_k_inverse():
    """The worst value for R k ranges from -1 to 0 depending on the distribution of the true labels."""
    C = np.random.randint(2, 11)
    golds = np.concatenate((np.arange(C), np.random.randint(0, C, size=np.random.randint(4, 100))), axis=0)
    preds = golds + 1 % C
    cm = ConfusionMatrix.create(golds, preds)
    assert -1.0 <= cm.get_r_k() <= 0.0
