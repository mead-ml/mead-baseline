import numpy as np
import time
import json
import argparse

from os import sys, path, makedirs
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from w2v import *
from data import *
from model import TaggerModel
from train import Trainer
from utils import revlut, fill_y
from torchy import long_0_tensor_alloc
DEF_BATCHSZ = 50
DEF_FILE_OUT = 'rnn-tagger'
DEF_EVAL_OUT = 'rnn-tagger-test.txt'
DEF_PATIENCE = 70
DEF_RNN = 'blstm'
DEF_NUM_RNN = 1
DEF_OPTIM = 'sgd'
DEF_EPOCHS = 1000
DEF_ETA = 0.001
DEF_HSZ = 100
DEF_CHARSZ = 16
DEF_WSZ = 30
DEF_CLIP = 5
DEF_DECAY = 0
DEF_MOM = 0.9
DEF_UNIF = 0.25
DEF_PDROP = 0.5
# By default, use max sentence length from data
DEF_MXLEN = -1
DEF_MXWLEN = 40
DEF_VALSPLIT = 0.15
DEF_EVAL_OUT = 'rnn-tagger-test.txt'
DEF_TEST_THRESH = 10

parser = argparse.ArgumentParser(description='Sequence tagger for sentences')

parser.add_argument('--eta', default=DEF_ETA , type=float)
parser.add_argument('--embed', default=None, help='Word2Vec embeddings file')
parser.add_argument('--cembed', default=None, help='Word2Vec char embeddings file')
parser.add_argument('--optim', default=DEF_OPTIM, help='Optim method')
parser.add_argument('--decay', default=DEF_DECAY, help='LR decay', type=float)
parser.add_argument('--mom', default=DEF_MOM, help='SGD momentum', type=float)
parser.add_argument('--dropout', default=DEF_PDROP, help='Dropout probability', type=float)
parser.add_argument('--train', help='Training file', required=True)
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file', required=True)
parser.add_argument('--rnn', default=DEF_RNN, help='RNN type')
parser.add_argument('--numrnn', default=DEF_NUM_RNN, help='The depth of stacked RNNs', type=int)
parser.add_argument('--outdir', default='out', help='Directory to put the output')
parser.add_argument('--conll_output', default=DEF_EVAL_OUT, help='Place to put test CONLL file')
parser.add_argument('--unif', default=DEF_UNIF, help='Initializer bounds for embeddings', type=float)
parser.add_argument('--clip', default=DEF_CLIP, help='Gradient clipping cutoff', type=float)
parser.add_argument('--epochs', default=DEF_EPOCHS, help='Number of epochs', type=int)
parser.add_argument('--batchsz', default=DEF_BATCHSZ, help='Batch size', type=int)
parser.add_argument('--mxlen', default=DEF_MXLEN, help='Max sentence length', type=int)
parser.add_argument('--mxwlen', default=DEF_MXWLEN, help='Max word length', type=int)
parser.add_argument('--cfiltsz', help='Filter sizes', nargs='+', default=[1,2,3,4,5,7], type=int)
parser.add_argument('--charsz', default=DEF_CHARSZ, help='Char embedding depth', type=int)
parser.add_argument('--patience', default=DEF_PATIENCE, help='Patience', type=int)
parser.add_argument('--hsz', default=DEF_HSZ, help='Hidden layer size', type=int)
parser.add_argument('--wsz', default=DEF_WSZ, help='Word embedding depth', type=int)
parser.add_argument('--valsplit', default=DEF_VALSPLIT, help='Validation split if no valid set', type=float)
parser.add_argument('--cbow', default=False, help='Do CBOW for characters', type=bool)
parser.add_argument('--nogpu', default=False, help='Use CPU (Not recommended)', type=bool)
parser.add_argument('--save', default=DEF_FILE_OUT, help='Save basename')
parser.add_argument('--fscore', default=0, help='Use F-score in metrics and early stopping', type=int)
parser.add_argument('--test_thresh', default=DEF_TEST_THRESH, help='How many epochs improvement required before testing', type=int)

args = parser.parse_args()
gpu = not args.nogpu

if path.exists(args.outdir) is False:
    print('Creating path: %s' % (args.outdir))
    makedirs(args.outdir)

maxs, maxw, vocab_ch, vocab_word = conll_build_vocab([args.train, 
                                                      args.test, 
                                                      args.valid])

maxw = min(maxw, args.mxwlen)
maxs = min(maxs, args.mxlen) if args.mxlen > 0 else maxs
print('Max sentence length %d' % maxs)
print('Max word length %d' % maxw)

# Vocab LUTs
word_vocab = None
char_vocab = None

if args.cbow is True:
    print('Using CBOW char embeddings')
    args.cfiltsz = 0
else:
    print('Using convolutional char embeddings')

word_vec = None
if args.embed:
    word_vec = w2v.Word2VecModel(args.embed, vocab_word, args.unif)
    word_vocab = word_vec.vocab

if args.cembed:
    print('Using pre-trained character embeddings ' + args.cembed)
    char_vec = w2v.Word2VecModel(args.cembed, vocab_ch, args.unif)
    char_vocab = char_vec.vocab

    args.charsz = char_vec.dsz
    if args.charsz != args.wsz and args.cbow is True:
        print('Warning, you have opted for CBOW char embeddings, and have provided pre-trained char vector embeddings.  To make this work, setting word vector size to character vector size %d' % args.charsz)
        args.wsz = args.charsz
else:
    if args.charsz != args.wsz and args.cbow is True:
        print('Warning, you have opted for CBOW char embeddings, but have provided differing sizes for char embedding depth and word depth.  This is not possible, forcing char embedding depth to be word depth ' + args.wsz)
        args.charsz = args.wsz

    char_vec = w2v.RandomInitVecModel(args.charsz, vocab_ch, args.unif)
    char_vocab = char_vec.vocab

f2i = {"<PAD>":0}

ts, f2i, _ = conll_load_sentences(args.train, word_vocab, char_vocab, maxs, maxw, f2i,  long_0_tensor_alloc)
print('Loaded  training data')

if args.valid is not None:
    print('Using provided validation data')
    vs, f2i,_ = conll_load_sentences(args.valid, word_vocab, char_vocab, maxs, maxw, f2i,  long_0_tensor_alloc)
else:
    ts, vs = valid_split(ts, args.valsplit)
    print('Created validation split')

es, f2i,txts = conll_load_sentences(args.test, word_vocab, char_vocab, maxs, maxw, f2i,  long_0_tensor_alloc)
print('Loaded test data')
i2f = revlut(f2i)
print(i2f)
print('Using %d examples for training' % len(ts))
print('Using %d examples for validation' % len(vs))
print('Using %d examples for test' % len(es))

model = TaggerModel(f2i, word_vec, char_vec, maxs, maxw, args.rnn, args.wsz, args.hsz,
                    args.cfiltsz, args.dropout, args.numrnn)
trainer = Trainer(gpu, model, args.optim, args.eta, args.mom)
outname = '%s/%s.model' % (args.outdir, args.save)
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

model = TaggerModel.load(args.outdir, args.save)
score = trainer.test(es, 1)
