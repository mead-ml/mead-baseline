"""Plot and compare the metrics from various calibrated models.

This script creates the following:

    * A csv file with columns for the Model Type (the label), and the various calibration metrics
    * A grid of graphs, the first row is confidence histograms for each model, the second row is
      the reliability diagram for that model.
    * If the problem as binary it creates calibration curves for each model all plotted on the same graph.

Matplotlib is required to use this script. The `tabulate` package is recommended but not required.

The input of this script is pickle files created by `$MEAD-BASELINE/api-examples/analyze_calibration.py`
"""

import csv
import pickle
import argparse
from collections import Counter
from eight_mile.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
    reliability_curve,
    confidence_histogram,
    Bins,
)
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Compare calibrated models by grouping visualizations and creating a table.")
parser.add_argument("--stats", nargs="+", default=[], required=True, help="A list of pickles created by the analyze_calibration.py script to compare.")
parser.add_argument("--labels", nargs="+", default=[], required=True, help="A list of labels to assign to each pickle, should have the same number of arguments as --stats")
parser.add_argument("--metrics-output", "--metrics_output", default="table.csv", help="Filename to save the resulting metrics into as a csv")
parser.add_argument("--curve-output", "--curve_output", default="curve.png", help="Filename to save the reliability curves graph to.")
parser.add_argument("--diagram-output", "--diagram_output", default="diagram.png", help="Filename to save the reliability diagrams and confidence histograms too.")
parser.add_argument("--figsize", default=10, type=int, help="The size of the figure, controls how tall the figure is.")
args = parser.parse_args()

# Make sure the labels and stats are aligned
if len(args.stats) != len(args.labels):
    raise ValueError(f"You need a label for each calibration stat you load. Got {len(args.stats)} stats and {len(args.labels)} labels")

# Make sure the labels are unique
counts = Counter(args.labels)
if any(v != 1 for v in counts.values()):
    raise ValueError(f"All labels must be unique, found duplicates of {[k for k, v in counts.items() if v != 1]}")

# Load the calibration stats
stats = []
for file_name in args.stats:
    with open(file_name, "rb") as f:
        stats.append(pickle.load(f))

# Make sure there is the same number of bins for each model
for field in stats[0]:
    if not isinstance(stats[0][field], Bins):
        continue
    lengths = []
    for stat in stats:
        if stat[field] is None:
            continue
        lengths.append(len(stat[field].accs))
    if len(set(lengths)) != 1:
        raise ValueError(f"It is meaningless to compare calibrations with different numbers of bins: Mismatch was found for {field}")


def get_metrics(data, model_type):
    return {
        "Model Type": model_type,
        "ECE": expected_calibration_error(data.accs, data.confs, data.counts) * 100,
        "MCE": maximum_calibration_error(data.accs, data.confs, data.counts) * 100,
    }

# Calculate the metrics based on the multiclass calibration bins
metrics = [get_metrics(stat['multiclass'], label) for stat, label in zip(stats, args.labels)]

# Print the metrics
try:
    # If you have tabulate installed it prints a nice postgres style table
    from tabulate import tabulate
    print(tabulate(metrics, headers="keys", floatfmt=".3f", tablefmt="psql"))
except ImportError:
    for metric in metrics:
        for k, v in metric.items():
            if isinstance(v, float):
                print(f"{k}: {v:.3f}")
            else:
                print(f"{k}: {v}")

# Write the metrics to a csv to look at later
with open(args.metrics_output, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=list(metrics[0].keys()), quoting=csv.QUOTE_MINIMAL, delimiter=",", dialect="unix")
    writer.writeheader()
    writer.writerows(metrics)

# Plot the histograms and graphs for each model
f, ax = plt.subplots(2, len(metrics), figsize=(args.figsize * len(metrics) // 2, args.figsize), sharey=True, sharex=True)
for i, (stat, label) in enumerate(zip(stats, args.labels)):
    # If you are the first model you get y_labels, everyone else just uses yours
    if i == 0:
        confidence_histogram(
            stat['histogram'].edges,
            stat['histogram'].counts,
            acc=stat['acc'],
            avg_conf=stat['conf'],
            title=f"{label}\nConfidence Distribution",
            x_label=None,
            ax=ax[0][i],
        )
        reliability_diagram(
            stat['multiclass'].accs,
            stat['multiclass'].confs,
            stat['multiclass'].edges,
            num_classes=stat['num_classes'],
            ax=ax[1][i]
        )
    else:
        confidence_histogram(
            stat['histogram'].edges,
            stat['histogram'].counts,
            acc=stat['acc'],
            avg_conf=stat['conf'],
            title=f"{label}\nConfidence Distribution",
            y_label=None,
            x_label=None,
            ax=ax[0][i],
        )
        reliability_diagram(
            stat['multiclass'].accs,
            stat['multiclass'].confs,
            stat['multiclass'].edges,
            num_classes=stat['num_classes'],
            y_label=None,
            ax=ax[1][i]
        )

f.savefig(args.diagram_output)
plt.show()

# Plot reliability curves for binary classification models
if stats[0]['num_classes'] == 2:
    f, ax = plt.subplots(1, 1, figsize=(args.figsize, args.figsize))
    for stat, label, color in zip(stats, args.labels, plt.rcParams['axes.prop_cycle'].by_key()['color']):
        reliability_curve(
            stat['binary'].accs,
            stat['binary'].confs,
            color=color,
            label=label,
            ax=ax
        )
    f.savefig(args.curve_output)
    plt.show()
