import baseline as bl
import baseline.tf.embeddings
import baseline.tf.classify
from eight_mile.tf.layers import set_tf_log_level
import time
import tensorflow as tf
import numpy as np
import os
import argparse
import logging
log = logging.getLogger('baseline.timing')

def get_logging_level(ll):
    ll = ll.lower()
    if ll == 'debug':
        return logging.DEBUG
    if ll == 'info':
        return logging.INFO
    return logging.WARNING


def get_tf_logging_level(ll):
    ll = ll.lower()
    if ll == 'debug':
        return tf.logging.DEBUG
    if ll == 'info':
        return logging.INFO
    return tf.logging.WARN

parser = argparse.ArgumentParser(description='Train a Baseline model with TensorFlow Estimator API')
parser.add_argument('--checkpoint_dir', help='Directory for model checkpoints', default='./checkpoints', type=str)
parser.add_argument('--model_type', help='What type of model to build', type=str, default='default')
parser.add_argument('--poolsz', help='How many hidden units for pooling', type=int, default=100)
parser.add_argument('--stacksz', help='How many hidden units for stacking', type=int, nargs='+')
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--backend', help='backend', default='tf')
parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) signature name', type=str)
parser.add_argument('--epochs', help='Number of epochs to train', type=int, default=2)
parser.add_argument('--batchsz', help='Batch size', type=int, default=50)
parser.add_argument('--filts', help='Parallel convolution filter widths (if default model)', type=int, default=[3, 4, 5], nargs='+')
parser.add_argument('--mxlen', help='Maximum post length (number of words) during training', type=int, default=100)
parser.add_argument('--train', help='Training file', default='../data/stsa.binary.phrases.train')
parser.add_argument('--valid', help='Validation file', default='../data/stsa.binary.dev')
parser.add_argument('--test', help='Testing file', default='../data/stsa.binary.test')
parser.add_argument('--embeddings', help='Pretrained embeddings file', default='/data/embeddings/GoogleNews-vectors-negative300.bin')
parser.add_argument('--ll', help='Log level', type=str, default='info')
parser.add_argument('--tf_ll', help='TensorFlow Log level', type=str, default='warning')
parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('--optim', help='Optimizer (sgd, adam) (default is adam)', type=str, default='adam')
args = parser.parse_known_args()[0]

logging.basicConfig(level=get_logging_level(args.ll))
set_tf_log_level(args.tf_ll)

pool_field = 'cmotsz' if args.model_type == 'default' else 'rnnsz'

model_params = {
    'model_type': args.model_type,
    'filtsz': args.filts,
    pool_field: args.poolsz
}

if args.stacksz is not None:
    model_params['hsz'] = args.stacksz


feature_desc = {
    'word': {
        'vectorizer': bl.Token1DVectorizer(mxlen=args.mxlen, transform_fn=bl.lowercase),
        'embed': {'file': args.embeddings, 'type': 'default', 'unif': 0.25}
    }
}
# Create a reader that is using our vectorizers to parse a TSV file
# with rows like:
# <label>\t<sentence>\n

vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}
reader = bl.TSVSeqLabelReader(vectorizers,
                              clean_fn=bl.TSVSeqLabelReader.do_clean)

train_file = args.train
valid_file = args.valid
test_file = args.test

# This builds a set of counters
vocabs, labels = reader.build_vocab([train_file,
                                     valid_file,
                                     test_file])

# This builds a set of embeddings objects, these are typically not DL-specific
# but if they happen to be addons, they can be
embeddings = dict()
for k, v in feature_desc.items():
    embed_config = v['embed']
    embeddings_for_k = bl.load_embeddings('word',
                                          embed_file=embed_config['file'],
                                          known_vocab=vocabs[k],
                                          embed_type=embed_config.get('type', 'default'),
                                          unif=embed_config.get('unif', 0.),
                                          use_mmap=True)

    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']

# Now create the model we want to train.  There are lots of HPs we can pass in
# but for this simple example, use basic defaults
model = bl.model.create_model(embeddings, labels, **model_params)

ts = reader.load(train_file, vocabs=vocabs, batchsz=args.batchsz)
vs = reader.load(valid_file, vocabs=vocabs, batchsz=args.batchsz)
es = reader.load(test_file, vocabs=vocabs, batchsz=args.batchsz)

bl.train.fit(model, ts, vs, es, epochs=args.epochs,
             optim=args.optim, eta=args.lr,
             reporting=[r.step for r in (bl.reporting.LoggingReporting(),)],
             fit_func='feed_dict')
