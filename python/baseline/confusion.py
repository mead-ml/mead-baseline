import numpy as np


class ConfusionMatrix:

    def __init__(self, labels):
        if type(labels) is dict:
            self.labels = []
            for i in range(len(labels)):
                self.labels.append(labels[i])
        else:
            self.labels = labels
        nc = len(self.labels)
        self._cm = np.zeros((nc, nc), dtype=np.int)

    def add(self, truth, guess):
        self._cm[truth, guess] += 1

    def __str__(self):
        values = []
        width = max(8, max(len(x) for x in self.labels) + 1)
        for i, label in enumerate([''] + self.labels):
            values += ["{:>{width}}".format(label, width=width+1)]
        values += ['\n']
        for i, label in enumerate(self.labels):
            values += ["{:>{width}}".format(label, width=width+1)]
            for j in range(len(self.labels)):
                values += ["{:{width}d}".format(self._cm[i, j], width=width + 1)]
            values += ['\n']
        values += ['\n']
        return ''.join(values)

    def reset(self):
        self._cm *= 0

    def get_correct(self):
        return self._cm.diagonal().sum()

    def get_total(self):
        return self._cm.sum()

    def get_acc(self):
        return self.get_correct()/self.get_total()

    def get_recall(self):
        total = np.sum(self._cm, axis=1) + 0.0000001
        return np.diag(self._cm) / total

    def get_precision(self):
        total =  np.sum(self._cm, axis=0) + 0.0000001
        return np.diag(self._cm) / total

    def get_mean_precision(self):
        return np.mean(self.get_precision())

    def get_mean_recall(self):
        return np.mean(self.get_recall())

    def get_macro_f(self, beta=1):
        p = self.get_mean_precision()
        r = self.get_mean_recall()
        if beta < 0:
            raise Exception('Beta must be greater than 0')
        return (beta*beta + 1) * p * r / (beta*beta * p + r)

    def get_f(self, beta=1):
        p = self.get_precision()[1]
        r = self.get_recall()[1]
        if beta < 0:
            raise Exception('Beta must be greater than 0')
        return (beta*beta + 1) * p * r / (beta*beta * p + r)

    def get_all_metrics(self):
        metrics = {'acc': self.get_acc()}
        # If 2 class, assume second class is positive AKA 1
        if len(self.labels) == 2:
            metrics['precision'] = self.get_precision()[1]
            metrics['recall'] = self.get_recall()[1]
            metrics['f1'] = self.get_f(1)
        else:
            metrics['mean_precision'] = self.get_mean_precision()
            metrics['mean_recall'] = self.get_mean_recall()
            metrics['macro_f1'] = self.get_macro_f(1)
        return metrics

    def add_batch(self, truth, guess):
        for truth_i, guess_i in zip(truth, guess):
            self.add(truth_i, guess_i)
