import numpy as np
import time
import argparse

from os import sys, path, makedirs
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from w2v import Word2VecModel
from reader import load_sentences, build_vocab
from data import DataFeed
from utils import revlut
from model import *
from torchy import *
from train import fit, show_examples

parser = argparse.ArgumentParser(description='Sequence tagger for sentences')

parser.add_argument('--eta', default=0.001, help='Initial learning rate.', type=float)
parser.add_argument('--mom', default=0.9, help='Momentum (if SGD)', type=float)
parser.add_argument('--embed1', help='Word2Vec embeddings file (1)', required=True)
parser.add_argument('--embed2', help='Word2Vec embeddings file (2)')
parser.add_argument('--rnntype', default='lstm', help='(lstm|gru)')
parser.add_argument('--optim', default='adam', help='Optim method')
parser.add_argument('--dropout', default=0.5, help='Dropout probability', type=float)
parser.add_argument('--train', help='Training file')
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file')
parser.add_argument('--unif', default=0.25, help='Initializer bounds for embeddings', type=float)
parser.add_argument('--epochs', default=60, help='Number of epochs', type=int)
parser.add_argument('--batchsz', default=50, help='Batch size', type=int)
parser.add_argument('--mxlen', default=100, help='Max length', type=int)
parser.add_argument('--patience', default=10, help='Patience', type=int)
parser.add_argument('--hsz', default=100, help='Hidden layer size', type=int)
parser.add_argument('--outfile', default='./seq2seq.pyth', help='Model file name')
parser.add_argument('--clip', default=5, help='Gradient clipping', type=float)
parser.add_argument('--layers', default=1, help='Number of LSTM layers for encoder/decoder', type=int)
parser.add_argument('--sharedv', default=False, help='Share vocab between source and destination', type=bool)
parser.add_argument('--showex', default=True, help='Show generated examples every few epochs', type=bool)
parser.add_argument('--sample', default=False, help='If showing examples, sample?', type=bool)
parser.add_argument('--topk', default=5, help='If sampling in examples, prunes to topk', type=int)
parser.add_argument('--attn', default=False, help='Use attention')
parser.add_argument('--max_examples', default=5, help='How many examples to show', type=int)
parser.add_argument('--nogpu', default=False, help='Dont use GPU (debug only!)', type=bool)
parser.add_argument('--pdrop', default=0.5, help='Dropout', type=float)

args = parser.parse_args()
gpu = not args.nogpu


f2i = {}
v1 = [0]
v2 = [1]

if args.sharedv is True:
    v1.append(1)
    v2.append(0)

vocab1 = build_vocab(v1, [args.train, args.test])
vocab2 = build_vocab(v2, [args.train, args.test])

embed1 = Word2VecModel(args.embed1, vocab1, args.unif)

print('Loaded word embeddings: ' + args.embed1)

if args.embed2 is None:
    print('No embed2 found, using embed1 for both')
    args.embed2 = args.embed1

embed2 = Word2VecModel(args.embed2, vocab2, args.unif)
print('Loaded word embeddings: ' + args.embed2)

tsx = load_sentences(args.train, embed1.vocab, embed2.vocab, args.mxlen, long_0_tensor_alloc)
esx = load_sentences(args.test, embed1.vocab, embed2.vocab, args.mxlen, long_0_tensor_alloc)

ts = DataFeed(tsx, args.batchsz, shuffle=True, alloc_fn=long_tensor_alloc, shape_fn=tensor_shape, src_trans_fn=None if args.attn else tensor_reverse_2nd, trim=True)
es = DataFeed(esx, args.batchsz, alloc_fn=long_tensor_alloc, shape_fn=tensor_shape, src_trans_fn=None if args.attn else tensor_reverse_2nd, trim=True)

rlut1 = revlut(embed1.vocab)
rlut2 = revlut(embed2.vocab)

Seq2SeqModelType = Seq2SeqAttnModel if args.attn else Seq2SeqModel
print(Seq2SeqModelType)
seq2seq = Seq2SeqModelType(embed1, embed2, args.hsz, args.layers, args.rnntype)

if args.showex:
    args.after_train_fn = lambda model: show_examples(gpu, model, es, rlut1, rlut2, embed2, args.mxlen, args.sample, args.topk, args.max_examples, rev=not args.attn)

fit(seq2seq, ts, es, **vars(args))
