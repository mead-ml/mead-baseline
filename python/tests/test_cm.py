import numpy as np
import pytest
from baseline import ConfusionMatrix

Y_TRUE = [2, 0, 2, 2, 0, 1, 3, 1, 3, 3, 3, 3, 4]
Y_PRED = [0, 0, 2, 2, 0, 2, 3, 3, 3, 1, 3, 2, 4]
LABELS = ['0', '1', '2', '3', '4']

CLASS_PREC = [0.666667, 0.0, 0.5, 0.75, 1.0]
CLASS_RECALL = [1.0, 0.0, 0.666667, 0.6, 1.0]
CLASS_F1 = [0.8, 0.0, 0.571429, 0.666667, 1.0]
CLASS_SUPPORT = [2, 2, 3, 5, 1]
TOL = 1e-6

def make_mc_cm():
    cm = ConfusionMatrix(LABELS)
    for y_t, y_p in zip(Y_TRUE, Y_PRED):
        cm.add(y_t, y_p)
    return cm

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
