import argparse
import codecs
import re
from baseline.progress import create_progress_bar


def read_lines(tsfile):
    txts = []
    lbls = []
    txt = []
    lbl = []
    with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
        for line in f:
            states = re.split("\s", line.strip())

            if len(states) > 1:
                txt.append(states[0])
                lbl.append(states[-1])
            else:
                txts.append(txt)
                lbls.append(lbl)
                txt = []
                lbl = []
    return txts, lbls


parser = argparse.ArgumentParser(description='Loads an RNNTaggerModel, predicts the labels for an input conll file and '
                                             'produces the output in the same format. The second column is the '
                                             'predicted label')
parser.add_argument('--inp', help='input conll', required=True)
parser.add_argument('--outp', help='output conll', required=True)
parser.add_argument('--model', help='model file: tagger-model-tf-*', required=True)
parser.add_argument('--mxlen', help='max. length of the sentence (provided during training)', type=int, required=True)
parser.add_argument('--mxwlen', help='max. length of a word (provided during training)', type=int, required=True)
parser.add_argument('--backend', choices=['tf', 'pytorch'], default='tf', help='Deep Learning Framework backend')
parser.add_argument('--modeltype', default='default', help='tagger model type')
# choices are ['default'] currently. default is RNNTaggerModel.
args = parser.parse_args()

if args.backend == 'tf':
    from baseline.tf.tagger.model import BASELINE_TAGGER_LOADERS
    tagger = BASELINE_TAGGER_LOADERS[args.modeltype](args.model)
else:
    from baseline.pytorch.tagger.model import BASELINE_TAGGER_LOADERS
    tagger = BASELINE_TAGGER_LOADERS[args.modeltype](args.model)

predlbls = []
inp_txts, gold_lbls = read_lines(args.inp)

pg = create_progress_bar(len(inp_txts))
with codecs.open(args.outp, encoding="utf-8", mode="w") as f:
    for index, sen in enumerate(inp_txts):
        pred_lbl_sen = [x[1] for x in tagger.predict_text(sen, mxlen=args.mxlen, maxw=args.mxwlen)]
        gold_lbl_sen = gold_lbls[index]
        for word, pred_lbl, gold_lbl in zip(sen, pred_lbl_sen, gold_lbl_sen):
            f.write("{} {} {}\n".format(word, pred_lbl, gold_lbl))
        f.write("\n")
        pg.update()

pg.done()
