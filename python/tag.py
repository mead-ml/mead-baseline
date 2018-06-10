import argparse
import codecs
import re
from baseline.progress import create_progress_bar
from baseline.utils import load_user_model, lowercase
from baseline.featurizers import create_featurizer 
import json
import numpy as np


def read_lines(tsfile):
    txts = []
    labels = []
    txt = []
    label = []
    with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
        for line in f:
            states = re.split("\s", line.strip())
            if len(states) > 1:
                txt.append(states[:-1])
                label.append(states[-1])
            else:
                txts.append(txt)
                labels.append(label)
                txt = []
                label = []
    return txts, labels


parser = argparse.ArgumentParser(description='Loads an RNNTaggerModel, predicts the labels for an input conll file and '
                                             'produces the output in the same format. The second column is the '
                                             'predicted label')
parser.add_argument('--input', help='input conll', required=True)
parser.add_argument('--output', help='output conll', required=True)
parser.add_argument('--model', help='model file: tagger-model-tf-*', required=True)
parser.add_argument('--mxlen', help='max. length of the sentence (provided during training)', type=int, required=True)
parser.add_argument('--mxwlen', help='max. length of a word (provided during training)', type=int, required=True)
parser.add_argument('--backend', choices=['tf', 'pytorch'], default='tf', help='Deep Learning Framework backend')
parser.add_argument('--features', default=None, help='JSON file with the feature name (must match with training config)'
                                                     'and the feature index in the CONLL file example: {"gaz":1}, when '
                                                     'the conll file has gazetteer feature in column 2')
parser.add_argument('--model_type', default='default', help='tagger model type')
parser.add_argument('--featurizer_type', default='default', help='featurizer type')

args = parser.parse_args()

if args.backend == 'tf':
    import baseline.tf.tagger.model as tagger
else:
    import baseline.pytorch.tagger.model as tagger
model = tagger.load_model(args.model, model_type=args.model_type)

predicted_labels = []
input_texts, gold_labels = read_lines(args.input)
vocab_keys = {'word': 0, 'char': None}

if args.features is not None:
    features = json.load(open(args.features))
    vocab_keys.update(features)


pg = create_progress_bar(len(input_texts))
featurizer = create_featurizer(model, args.mxlen, args.mxwlen, zero_alloc=np.zeros,
                               vocab_keys=vocab_keys, featurizer_type=args.featurizer_type)
with codecs.open(args.output, encoding="utf-8", mode="w") as f:
    for index, sen in enumerate(input_texts):
        predicted_label_sen = [x[1] for x in model.predict_text(sen, mxlen=args.mxlen, maxw=args.mxwlen,
                                                                featurizer=featurizer, word_trans_fn=lowercase)]
        gold_label_sen = gold_labels[index]
        for word_feature, predicted_label, gold_label in zip(sen, predicted_label_sen, gold_label_sen):
            f.write("{} {} {}\n".format(" ".join(word_feature), gold_label, predicted_label))
        f.write("\n")
        pg.update()

