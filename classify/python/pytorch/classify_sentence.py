import numpy as np
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import ConvModel
from train import Trainer
from os import sys, path, makedirs
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from w2v import *
from data import *
from utils import revlut, mdsave
from torchy import long_0_tensor_alloc, TorchExamples
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--eta', help='Initial learning rate', default=0.001, type=float)
parser.add_argument('--mom', help='SGD Momentum', default=0.9, type=float)
parser.add_argument('--embed', help='Word2Vec embeddings file', required=True)
parser.add_argument('--train', help='Training file', required=True)
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file', required=True)
parser.add_argument('--save', help='Save basename', default='classify_sentence_pytorch')
parser.add_argument('--nogpu', help='Do not use GPU', default=False)
parser.add_argument('--optim', help='Optim method', default='adam', choices=['adam', 'adagrad', 'adadelta', 'sgd'])
parser.add_argument('--dropout', help='Dropout probability', default=0.5, type=float)
parser.add_argument('--unif', help='Initializer bounds for embeddings', default=0.25)
parser.add_argument('--epochs', help='Number of epochs', default=25, type=int)
parser.add_argument('--batchsz', help='Batch size', default=50, type=int)
parser.add_argument('--mxlen', help='Max length', default=100, type=int)
parser.add_argument('--patience', help='Patience', default=10, type=int)
parser.add_argument('--cmotsz', help='Hidden layer size', default=100, type=int)
parser.add_argument('--hsz', help='Projection layer size', default=-1, type=int)
parser.add_argument('--filtsz', help='Filter sizes', nargs='+', default=[3,4,5], type=int)
parser.add_argument('--clean', help='Do cleaning', action='store_true')
parser.add_argument('--static', help='Fix pre-trained embeddings weights', action='store_true')
parser.add_argument('--chars', help='Use characters instead of words', action='store_true')
parser.add_argument('--valsplit', help='Validation split if no valid set', default=0.15, type=float)
parser.add_argument('--outdir', help='Output directory', default='./train')

args = parser.parse_args()
gpu = not args.nogpu


if path.exists(args.outdir) is False:
    print('Creating path: %s' % (args.outdir))
    makedirs(args.outdir)
vocab = build_vocab([args.train, args.test, args.valid], args.clean, args.chars)


unif = 0 if args.static else args.unif
embeddings = Word2VecModel(args.embed, vocab, unif)

mxfiltsz = np.max(args.filtsz)
f2i = {}
ts, f2i = load_sentences(args.train, embeddings.vocab, f2i, args.clean, args.chars, args.mxlen, mxfiltsz, vec_alloc=long_0_tensor_alloc, ExType=TorchExamples)
print('Loaded training data')

valsplit = args.valsplit
valdata = None
if args.valid is not None:
    print('Using provided validation data')
    valsplit = 0
    vs, f2i = load_sentences(args.valid, embeddings.vocab, f2i, args.clean, args.chars, args.mxlen, mxfiltsz, vec_alloc=long_0_tensor_alloc, ExType=TorchExamples)
else:
    ts, vs = valid_split(ts, args.valsplit, ExType=TorchExamples)

es, f2i = load_sentences(args.test, embeddings.vocab, f2i, args.clean, args.chars, args.mxlen, mxfiltsz, vec_alloc=long_0_tensor_alloc, ExType=TorchExamples)
print('Loaded test data')

nc = len(f2i)


mdsave(f2i, embeddings.vocab, args.outdir, args.save)

model = ConvModel(embeddings, nc, args.filtsz, args.cmotsz, args.hsz, args.dropout, not args.static)
trainer = Trainer(gpu, model, args.optim, args.eta, args.mom)

max_acc = 0
last_improved = 0

for i in range(args.epochs):
    print('Training epoch %d' % (i+1))
    trainer.train(ts, args.batchsz)
    this_acc = trainer.test(vs, args.batchsz, 'Validation')
    if this_acc > max_acc:
        max_acc = this_acc
        last_improved = i
        model.save(args.outdir, args.save)
        print('Highest dev acc achieved yet -- writing model')

    if (i - last_improved) > args.patience:
        print('Stopping due to persistent failures to improve')
        break

            
print("-----------------------------------------------------")
print('Highest dev acc %.2f' % (max_acc * 100.))
print('=====================================================')

print('=====================================================')
print('Evaluating best model on test data:')
print('=====================================================')
model = ConvModel.load(args.outdir, args.save)

score = trainer.test(es, 2)
