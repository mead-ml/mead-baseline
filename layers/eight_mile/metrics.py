import numpy as np
from eight_mile.utils import exporter, Offsets


__all__ = []
export = exporter(__all__)


@export
# Metrics UAS/LAS
# UCM (unlabeled complete matching rate, the % of sentences with whole correct trees
# LCM labeled complete matching rate
class LAS:
    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, heads_pred, heads_gold, labels_pred, labels_gold):
        valid = labels_gold != Offsets.PAD
        for h_p, h_g, l_p, l_g in zip(heads_pred[valid], heads_gold[valid], labels_pred[valid], labels_gold[valid]):
            correct = h_p == h_g and l_p == l_g
            self.total += 1
            self.correct += float(correct)

    @property
    def name(self):
        return 'las'

    @property
    def score(self):
        return self.correct / self.total


@export
class UAS:
    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, heads_pred, heads_gold, labels_pred, labels_gold):
        valid = labels_gold != Offsets.PAD
        for h_p, h_g in zip(heads_pred[valid], heads_gold[valid]):
            correct = h_p == h_g
            self.total += 1
            self.correct += float(correct)

    @property
    def name(self):
        return 'uas'

    @property
    def score(self):
        return self.correct / self.total


@export
class LCM:
    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, heads_pred, heads_gold, labels_pred, labels_gold):
        valid = labels_gold != Offsets.PAD
        correct = True
        # For LCM, the entire tree has to match with labels
        for h_p, h_g, l_p, l_g in zip(heads_pred[valid], heads_gold[valid], labels_pred[valid], labels_gold[valid]):
            correct = correct and (h_p == h_g and l_p == l_g)

        self.total += 1
        self.correct += int(correct)

    @property
    def name(self):
        return 'ucm'

    @property
    def score(self):
        return self.correct / self.total


@export
class UCM:
    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, heads_pred, heads_gold, labels_pred, labels_gold):
        valid = labels_gold != Offsets.PAD
        correct = True
        # For UCM, the entire tree has to match
        for h_p, h_g in zip(heads_pred[valid], heads_gold[valid]):
            correct = correct and (h_p == h_g)

        self.total += 1
        self.correct += int(correct)

    @property
    def name(self):
        return 'ucm'

    @property
    def score(self):
        return self.correct / self.total
