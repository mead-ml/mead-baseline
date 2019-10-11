import argparse
import baseline
from eight_mile.utils import get_version
from eight_mile.confusion import ConfusionMatrix
import eight_mile.tf.embeddings
import eight_mile.tf.layers as L
from eight_mile.tf.layers import TRAIN_FLAG, SET_TRAIN_FLAG
from eight_mile.tf.optz import EagerOptimizer
import tensorflow as tf
import logging
import numpy as np
import time



TF_VERSION = get_version(tf)
if TF_VERSION < 2:
    from tensorflow import count_nonzero
    tf.enable_eager_execution()
    Adam = tf.train.AdamOptimizer

else:
    from tensorflow.compat.v1 import count_nonzero
    Adam = tf.optimizers.Adam

#tf.config.gpu.set_per_process_memory_growth(True)

NUM_PREFETCH = 2
SHUF_BUF_SZ = 5000


def get_logging_level(ll):
    ll = ll.lower()
    if ll == 'debug':
        return logging.DEBUG
    if ll == 'info':
        return logging.INFO
    return logging.WARNING


parser = argparse.ArgumentParser(description='Train a Layers model with TensorFlow API')
parser.add_argument('--model_type', help='What type of model to build', type=str, default='default')
parser.add_argument('--poolsz', help='How many hidden units for pooling', type=int, default=100)
parser.add_argument('--stacksz', help='How many hidden units for stacking', type=int, nargs='+')
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
parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)

args = parser.parse_known_args()[0]


def to_tensors(ts):
    X = []
    y = []
    for sample in ts:
        X.append(sample['word'].squeeze())
        y.append(sample['y'].squeeze())
    return np.stack(X), np.stack(y)

feature_desc = {
    'word': {
        'vectorizer': baseline.Token1DVectorizer(mxlen=100, transform_fn=baseline.lowercase),
        'embed': {'file': args.embeddings, 'type': 'default', 'unif': 0.25}
    }
}

vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}
reader = baseline.TSVSeqLabelReader(vectorizers, clean_fn=baseline.TSVSeqLabelReader.do_clean)

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
    embeddings_for_k = eight_mile.embeddings.load_embeddings('word', embed_file=embed_config['file'], known_vocab=vocabs[k],
                                                             embed_type=embed_config.get('type', 'default'),
                                                             unif=embed_config.get('unif', 0.), use_mmap=True)

    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']


X_train, y_train = to_tensors(reader.load(train_file, vocabs=vocabs, batchsz=1))
X_valid, y_valid = to_tensors(reader.load(valid_file, vocabs=vocabs, batchsz=1))
X_test, y_test = to_tensors(reader.load(test_file, vocabs=vocabs, batchsz=1))

stacksz = len(args.filts) * args.poolsz
num_epochs = 2


def train_input_fn():
    SET_TRAIN_FLAG(True)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    dataset = dataset.batch(50)
    #dataset = dataset.map(lambda x, y: (x, y))
    dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': count_nonzero(x, axis=1)}, y))
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(NUM_PREFETCH)
    #_ = dataset.make_one_shot_iterator()
    return dataset


def valid_input_fn():
    SET_TRAIN_FLAG(False)
    dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    dataset = dataset.batch(50)
    dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': count_nonzero(x, axis=1)}, y))
    #dataset = dataset.map(lambda x, y: (x, y))
    #_ = dataset.make_one_shot_iterator()

    return dataset


def test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': count_nonzero(x, axis=1)}, y))
    #_ = dataset.make_one_shot_iterator()
    return dataset


model = L.EmbedPoolStackModel(2, embeddings, L.ParallelConv(300, args.poolsz, args.filts), L.Highway(stacksz))
#model = L.EmbedPoolStackModel(2, embeddings, L.LSTMEncoderHidden(100, 1), L.Highway(100))

def loss(model, x, y):
  y_ = model(x)
  return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


# 2.0 function
optimizer = EagerOptimizer(loss, Adam(0.001))

num_epochs = 2


for epoch in range(num_epochs):
    loss_acc = 0.
    step = 0
    start = time.time()
    for x, y in train_input_fn():
        loss_value = optimizer.update(model, x, y)
        loss_acc += loss_value
        step += 1
    print('training time {}'.format(time.time() - start))
    mean_loss = loss_acc / step
    print('Training Loss {}'.format(mean_loss))
    cm = ConfusionMatrix(['0', '1'])
    for x, y in valid_input_fn():
        y_ = np.argmax(model(x), axis=1)
        cm.add_batch(y, y_)
    print(cm)
    print(cm.get_all_metrics())

print('FINAL')
cm = ConfusionMatrix(['0', '1'])
for x, y in test_input_fn():
    # Optimize the model

    # Track progress
    # compare predicted label to actual label
    y_ = tf.argmax(model(x), axis=1, output_type=tf.int32)
    cm.add_batch(y, y_)

print(cm)
print(cm.get_all_metrics())
