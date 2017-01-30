from keras.models import Model, load_model
from keras.layers import Dense, Activation, Convolution1D, Embedding, Input, merge, GlobalMaxPooling1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import create_model
import numpy as np
from keras.utils import np_utils
from os import sys, path, makedirs
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from w2v import *
from data import *
import time
import json
import argparse
from utils import revlut, mdsave

parser = argparse.ArgumentParser(description='CNN classification model for sentences')
parser.add_argument('--embed', help='Word2Vec embeddings file', required=True)
parser.add_argument('--train', help='Training file', required=True)
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file', required=True)
parser.add_argument('--save', help='Save basename', default='classify_sentence_keras')
parser.add_argument('--optim', help='Optim method', default='adam', choices=['adam', 'adagrad', 'adadelta', 'sgd', 'rmsprop'])
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

if path.exists(args.outdir) is False:
    print('Creating path: %s' % (args.outdir))
    makedirs(args.outdir)
vocab = build_vocab([args.train, args.test, args.valid], args.clean, args.chars)


unif = 0 if args.static else args.unif
embeddings = Word2VecModel(args.embed, vocab, unif)

mxfiltsz = np.max(args.filtsz)
f2i = {}
ts, f2i = load_sentences(args.train, embeddings.vocab, f2i, args.clean, args.chars, args.mxlen, mxfiltsz)
print('Loaded  training data')

valsplit = args.valsplit
valdata = None
if args.valid is not None:
    print('Using provided validation data')
    valsplit = 0
    vs, f2i = load_sentences(args.valid, embeddings.vocab, f2i, args.clean, args.chars, args.mxlen, mxfiltsz)

es, f2i = load_sentences(args.test, embeddings.vocab, f2i, args.clean, args.chars, args.mxlen, mxfiltsz)
print('Loaded test data')

nc = len(f2i)


mdsave(f2i, embeddings.vocab, args.outdir, args.save)

model = create_model(embeddings, nc, args.filtsz, args.cmotsz, args.hsz, args.mxlen, args.dropout, not args.static)

model.compile(args.optim, 'categorical_crossentropy' , metrics=['accuracy'])

ts.y = np_utils.to_categorical(ts.y, nc)
es.y = np_utils.to_categorical(es.y, nc)
if args.valid is not None:
    vs.y = np_utils.to_categorical(vs.y, nc)

early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1, mode='auto')

outname = '%s/%s.model' % (args.outdir, args.save)
checkpoint = ModelCheckpoint(outname, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')

model.fit(ts.x, ts.y, args.batchsz, args.epochs, verbose=1, callbacks=[checkpoint, early_stopping], validation_split=valsplit, validation_data=(vs.x, vs.y), shuffle=True)

print('=====================================================')
print('Evaluating best model on test data:')
print('=====================================================')
model = load_model(outname)
start_time = time.time()

score = model.evaluate(es.x, es.y, args.batchsz, verbose=1)
duration = time.time() - start_time

print('Test (Loss %.4f) (Acc = %.4f) (%.3f sec)' % (score[0], score[1], duration))
