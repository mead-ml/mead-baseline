import argparse
import codecs
from baseline.progress import create_progress_bar
from baseline.reader import TSVSeqLabelReader


def classify_batch(model, data_batch):
    outcomes = [sorted(outcome, key=lambda tup: tup[1], reverse=True)[0] for outcome in model.classify(data_batch)]
    return [x[0] for x in outcomes]


parser = argparse.ArgumentParser(description='Loads a classifier model, predicts the labels for an input csv file and '
                                             'produces the output in the same format. The first column is the '
                                             'gold label, the second column is the predicted one')
parser.add_argument('--inp', help='input csv', required=True)
parser.add_argument('--outp', help='output csv', required=True)
parser.add_argument('--model', help='model file, eg: classify-model-tf-*', required=True)
parser.add_argument('--mxlen', help='max. length of the sentence (provided during training)', type=int, required=True)
parser.add_argument('--backend', choices=['tf', 'pytorch'], default='tf', help='Deep Learning Framework backend')
parser.add_argument('--modeltype', default='default', help='classifier model type')
# choices are ['default', 'lstm', 'nbow', 'nbowmax'] currently. default is ConvModel

args = parser.parse_args()
print("loading model...")
if args.backend == 'tf':
    from baseline.tf.classify.model import BASELINE_CLASSIFICATION_LOADERS
    classifier = BASELINE_CLASSIFICATION_LOADERS[args.modeltype](args.model)
else:
    from baseline.pytorch.classify.model import BASELINE_CLASSIFICATION_LOADERS
    classifier = BASELINE_CLASSIFICATION_LOADERS[args.modeltype](args.model)
print("loading data...")
reader = TSVSeqLabelReader(mxlen=args.mxlen)
vocab, labels = reader.build_vocab([args.inp])
word_vocab = classifier.get_vocab()
data = reader.load(args.inp, {'word': word_vocab}, batchsz=60, shuffle=False)
print("data loaded")
pg = create_progress_bar(len(data))
with codecs.open(args.outp, encoding="utf-8", mode="w") as f_out:
    for index, data_batch in enumerate(data):
        pred_label_batch = classify_batch(classifier, data_batch)
        f_out.write("{}\t{}\t{}\n".format(gold_label, pred_label, sen))
        pg.update()
pg.done()
