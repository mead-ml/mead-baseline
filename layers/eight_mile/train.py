from typing import Optional, Dict, List, Tuple
from eight_mile.utils import listify, span_f1
from eight_mile.confusion import ConfusionMatrix
from eight_mile.utils import Average, get_num_gpus_multiworker


class Metric:
    def reduce(self):
        pass


class AverageMetric(Average, Metric):

    def reduce(self):
        return self.avg


class SpanF1Metric(Metric):

    def __init__(self, val=None):
        self.gold_spans = []
        self.pred_spans = []
        if val:
            self.update(val)

    def update(self, val):
        if isinstance(val, tuple):
            golds, preds = val
            self.gold_spans += golds
            self.pred_spans += preds
        else:
            self.gold_spans += val.gold_spans
            self.pred_spans += val.pred_spans


    def reduce(self):
        return span_f1(self.gold_spans, self.pred_spans)


class GlobalMetrics:

    def __init__(self):
        self.metrics = {}

    def reduce(self):
        metrics = {}
        for metric in self.metrics.keys():
            if isinstance(self.metrics[metric], ConfusionMatrix):
                all_metrics = self.metrics[metric].get_all_metrics()
                for cm_metric in all_metrics:
                    metrics[cm_metric] = all_metrics[cm_metric]

            else:
                metrics[metric] = self.metrics[metric].reduce()
        return metrics


    def update(self, local_metrics):
        for metric, v in local_metrics.items():
            if metric not in self.metrics:
                if isinstance(v, ConfusionMatrix):
                    self.metrics[metric] = ConfusionMatrix(v.labels)
                elif isinstance(v, Metric):
                    self.metrics[metric] = v.__class__()
                else:
                    self.metrics[metric] = AverageMetric(metric)

            if isinstance(v, tuple):
                # k-out-of-N
                denom = v[1] if v[1] > 0 else 1
                self.metrics[metric].update(v[0]/denom, n=denom)
            else:
                self.metrics[metric].update(v)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __len__(self):
        return len(self.metrics)

    def keys(self):
        return self.metrics.keys()

    def items(self):
        return self.metrics.items()

    def values(self):
        return self.metrics.values()


class MetricObserver:

    def run(self, model, metrics, global_step):
        pass


class LogAllMetrics(MetricObserver):

    def __init__(self, name):
        self.name = name

    def run(self, model, metrics, global_step):
        name = f"[{self.name}]"
        print(f"\n{name:>14}: {global_step}")
        for metric in metrics.keys():
            print(f"{metric:>14}: {metrics[metric]}")
