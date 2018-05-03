from baseline.tf.tagger import RNNTaggerModel
import argparse
import codecs
import re


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

args = parser.parse_args()

tagger = RNNTaggerModel.load(args.model)
print("model loaded")

predlbls = []
inptxts, goldlbls = read_lines(args.inp)

with codecs.open(args.outp, encoding="utf-8", mode="w") as f:
    for index, sen in enumerate(inptxts):
        predlbl_sen = [x[1] for x in tagger.predict_text(sen, mxlen=args.mxlen, maxw=args.mxwlen)]
        goldlbl_sen = goldlbls[index]
        for word, predlbl, goldlbl in zip(sen, predlbl_sen, goldlbl_sen):
            f.write("{} {} {}\n".format(word, predlbl, goldlbl))
        f.write("\n")
        if not index % 100:
            print("{} percent processed".format(int((index/len(inptxts))*100)))
