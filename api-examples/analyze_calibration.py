"""Calculate and plot metrics on calibration like in https://arxiv.org/abs/1706.04599

While not needed having matplotlib installed gives nicer results.
"""

import os
import re
import pickle
import argparse
import numpy as np
import baseline as bl
from eight_mile.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    multiclass_calibration_bins,
    binary_calibration_bins,
    average_confidence,
)


parser = argparse.ArgumentParser(description="Analyze the calibration of a classifier")
parser.add_argument('--model', help='The path to either the .zip file created by training or to the client bundle created by exporting', required=True, type=str)
parser.add_argument('--backend', help='The deep learning backend your model was trained with', choices={'tf', 'pytorch'}, default='tf')
parser.add_argument('--device', help='The device to run your model on')
parser.add_argument('--batchsz', help='The number of examples to run at once', default=100, type=int)
parser.add_argument('--data', help="The data to test calibration on in the label first format", required=True)
parser.add_argument('--bins', help="The number of bins to use in calibration", default=10, type=int)
parser.add_argument('--hist-bins', '--hist_bins', help="The number of bins to use when creating the confidence histogram", default=100, type=int)
parser.add_argument('--output', help="The name of the output pickle that holds calibration stats")
parser.add_argument('--prefer_eager', help="If running in TensorFlow, should we prefer eager model", type=bl.str2bool, default=False)
args = parser.parse_args()

if args.backend == 'tf':
    from eight_mile.tf.layers import set_tf_eager_mode
    set_tf_eager_mode(args.prefer_eager)

# Read in the dataset
labels, texts = bl.read_label_first_data(args.data)

# Batch the dataset and the labels
batched = [texts[i : i + args.batchsz] for i in range(0, len(texts), args.batchsz)]
batched_labels = [labels[i : i + args.batchsz] for i in range(0, len(labels), args.batchsz)]

# Load the model
m = bl.ClassifierService.load(args.model, backend=args.backend, device=args.device)
# Extract the label vocab from the model: Note this is p messy and we should have a better story about handling label vocabs
label_vocab = {l: i for i, l in enumerate(m.get_labels())}

# Process batches to collect the probabilities and gold labels for each example
probs = []
labels = []
for texts, labs in zip(batched, batched_labels):
    prob = m.predict(texts, dense=True)
    probs.append(bl.to_numpy(prob))
    labels.extend(label_vocab[l] for l in labs)

probs = np.concatenate(probs, axis=0)
labels = np.array(labels)

# Binning
bins = multiclass_calibration_bins(labels, probs, bins=args.bins)
hist_bins = multiclass_calibration_bins(labels, probs, bins=args.hist_bins)
binary_bins = binary_calibration_bins(labels, probs[:, 1], bins=args.bins) if len(label_vocab) == 2 else None
acc = np.mean(labels == np.argmax(probs, axis=1))
conf = average_confidence(probs)

# Metrics
print(f"ECE: {expected_calibration_error(bins.accs, bins.confs, bins.counts)}")
print(f"MCE: {maximum_calibration_error(bins.accs, bins.confs, bins.counts)}")

# Save the needed data to a pickle so we can recreate the graphs if we need to
args.output = f"{re.sub(r'.pkl$', '', args.output)}.pkl" if args.output is not None else f"{args.model}-calibration-stats.pkl"
with open(args.output, "wb") as wf:
    pickle.dump({
        "multiclass": bins,
        "histogram": hist_bins,
        "binary": binary_bins,
        "acc": acc,
        "conf": conf,
        "num_classes": len(label_vocab)
    }, wf)

try:
    # If they have matplotlib installed plot the reliability graphs and the confidence histograms
    import matplotlib.pyplot as plt
    from eight_mile.calibration import reliability_diagram, reliability_curve, confidence_histogram
    reliability_diagram(
        bins.accs,
        bins.confs,
        bins.edges,
        num_classes=len(label_vocab),
        title=f"Reliability Diagram\n{args.model}"
    )
    confidence_histogram(
        hist_bins.edges,
        hist_bins.counts,
        acc=acc,
        avg_conf=conf,
        x_ticks=np.arange(0, 1.01, 0.2),
        title=f"Confidence Histogram\n{args.model}"
    )
    # If we using a binary classifier look at the reliability curve a la sklearn
    if binary_bins:
        reliability_curve(binary_bins.accs, binary_bins.confs, title=f"Reliability Curve\n{args.model}")
    plt.show()
except ImportError:
    pass
