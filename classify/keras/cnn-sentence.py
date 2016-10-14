# For Keras impl. of non-fine-tuned embedding, I just "froze" the embedding 
# layer, so if you want to see how to load and use full word2vec models
# by expanding each into a temporal continuous tensor, take a look at the 
# Tensorflow version or the Torch one.
#
# Other than that one line, change, only the model names differ from 
# cnn-sentence-fine here

from keras.models import Model, load_model
from keras.layers import Dense, Activation, Convolution1D, Embedding, Input, merge, GlobalMaxPooling1D, Dropout
#from keras.utils.visualize_util import plot, model_to_dot
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from keras.utils import np_utils
import w2v
import time
import json
import argparse
from data import buildVocab
from data import loadTemporalIndices
from utils import revlut
import os.path
import os

# We need to keep around our vector maps to preserve lookups of words
def mdsave(labels, vocab, outdir):
    basename = '%s/cnn-sentence' % outdir
    
    label_file = basename + '.labels'
    print("Saving attested labels '%s'" % label_file)
    with open(label_file, 'w') as f:
        json.dump(labels, f)

    vocab_file = basename + '.vocab'
    print("Saving attested vocabulary '%s'" % vocab_file)
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)


# Use the functional API since we support parallel convolutions
def create(embeddings, nc, filtsz, cmotsz, hsz, maxlen, pdrop):
    x = Input(shape=(maxlen,), dtype='int32', name='input')

    vocab_size = embeddings.weights.shape[0]
    embedding_dim = embeddings.dsz
    
    lut = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embeddings.weights], input_length=maxlen, trainable=False)
    
    embed = lut(x)

    mots = []
    for i, fsz in enumerate(filtsz):
        conv = Convolution1D(cmotsz, fsz, activation='relu', input_length=maxlen)(embed)
        gmp = GlobalMaxPooling1D()(conv)
        mots.append(gmp)

    joined = merge(mots, mode='concat')
    cmotsz_all = cmotsz * len(filtsz)
    drop1 = Dropout(pdrop)(joined)

    input_dim = cmotsz_all
    last_layer = drop1

    if hsz > 0:
        proj = Dense(hsz, input_dim=cmotsz_all, activation='relu')(drop1)
        drop2 = Dropout(pdrop)(proj)
        input_dim = hsz
        last_layer = drop2

    dense = Dense(output_dim=nc, input_dim=input_dim, activation='softmax')(last_layer)
    model = Model(input=[x], output=[dense])
#    plot(model, 'model.png')
    return model

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--eta', help='Initial learning rate', default=0.001, type=float)
parser.add_argument('--embed', help='Word2Vec embeddings file', required=True)
parser.add_argument('--train', help='Training file', required=True)
parser.add_argument('--valid', help='Validation file')
parser.add_argument('--test', help='Test file', required=True)


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
parser.add_argument('--chars', help='Use characters instead of words', action='store_true')
parser.add_argument('--valsplit', help='Validation split if no valid set', default=0.15, type=float)
parser.add_argument('--outdir', help='Output directory', default='./train')

args = parser.parse_args()

if os.path.exists(args.outdir) is False:
    print('Creating path: %s' % (args.outdir))
    os.makedirs(args.outdir)
vocab = buildVocab([args.train, args.test, args.valid], args.clean, args.chars)

embeddings = w2v.Word2VecModel(args.embed, vocab, args.unif)


mxfiltsz = np.max(args.filtsz)
f2i = {}
X_train, y_train, f2i = loadTemporalIndices(args.train, embeddings.vocab, f2i, args.clean, args.chars, args.mxlen, mxfiltsz)
print('Loaded  training data')

valsplit = args.valsplit
valdata = None
if args.valid is not None:
    print('Using provided validation data')
    valsplit = 0
    X_valid, y_valid, f2i = loadTemporalIndices(args.valid, embeddings.vocab, f2i, args.clean, args.chars, args.mxlen, mxfiltsz)
    valdata = [X_valid, y_valid]

X_test, y_test, f2i = loadTemporalIndices(args.test, embeddings.vocab, f2i, args.clean, args.chars, args.mxlen, mxfiltsz)
print('Loaded test data')

nc = len(f2i)


mdsave(f2i, embeddings.vocab, args.outdir)

model = create(embeddings, nc, args.filtsz, args.cmotsz, args.hsz, args.mxlen, args.dropout)

model.compile(args.optim, 'categorical_crossentropy' , metrics=['accuracy'])

y_train = np_utils.to_categorical(y_train, nc)
y_test = np_utils.to_categorical(y_test, nc)
if args.valid is not None:
    valdata[1] = np_utils.to_categorical(valdata[1], nc)

early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1, mode='auto')

checkpoint = ModelCheckpoint('%s/cnn-sentence.model' % args.outdir, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
model.fit(X_train, y_train, args.batchsz, args.epochs, verbose=1, callbacks=[checkpoint, early_stopping], validation_split=valsplit, validation_data=valdata, shuffle=True)

print('=====================================================')
print('Evaluating best model on test data:')
print('=====================================================')
model = load_model('%s/cnn-sentence.model' % args.outdir)
start_time = time.time()

score = model.evaluate(X_test, y_test, args.batchsz, verbose=1)
duration = time.time() - start_time

print('Test (Loss %.4f) (Acc = %.4f) (%.3f sec)' % (score[0], score[1], duration))
