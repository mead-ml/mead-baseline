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


def to_device(m):
    return m


def to_host(o):
    return o


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


class Data:

    def __init__(self, ts, batchsz):
        self.x, self.y = self._to_tensors(ts)
        self.batchsz = batchsz

    def _to_tensors(self, ts):
        x = []
        y = []
        for sample in ts:
            x.append(sample['word'].squeeze())
            y.append(sample['y'].squeeze())
        return np.stack(x), np.stack(y)

    def get_input(self, training=False):
        SET_TRAIN_FLAG(training)
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        dataset = dataset.shuffle(buffer_size=SHUF_BUF_SZ)
        dataset = dataset.batch(50)
        dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': count_nonzero(x, axis=1)}, y))
        dataset = dataset.prefetch(NUM_PREFETCH)
        return dataset


train_set = Data(reader.load(train_file, vocabs=vocabs, batchsz=1), args.batchsz)
valid_set = Data(reader.load(valid_file, vocabs=vocabs, batchsz=1), args.batchsz)
test_set = Data(reader.load(test_file, vocabs=vocabs, batchsz=1), args.batchsz)

stacksz = len(args.filts) * args.poolsz
num_epochs = 2

model = to_device(
    L.EmbedPoolStackModel(2, embeddings, L.ParallelConv(300, args.poolsz, args.filts), L.Highway(stacksz))
)


def loss(model, x, y):
  y_ = model(x)
  return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


# This works with TF 2.0 and PyTorch:
#optimizer = EagerOptimizer(loss, optim="adam", lr=0.001)

# This works on all 3
optimizer = EagerOptimizer(loss, Adam(0.001))

for epoch in range(num_epochs):
    loss_acc = 0.
    step = 0
    start = time.time()
    for x, y in train_set.get_input(training=True):
        loss_value = optimizer.update(model, x, y)
        loss_acc += loss_value
        step += 1
    
    print('training time {}'.format(time.time() - start))
    mean_loss = loss_acc / step
    print('Training Loss {}'.format(mean_loss))
    cm = ConfusionMatrix(['0', '1'])
    for x, y in valid_set.get_input():
        y_ = np.argmax(to_device(model(x)), axis=1)
        cm.add_batch(y, y_)
    print(cm)
    print(cm.get_all_metrics())

print('FINAL')
cm = ConfusionMatrix(['0', '1'])
for x, y in test_set.get_input():
    y_ = tf.argmax(to_device(model(x)), axis=1, output_type=tf.int32)
    cm.add_batch(y, y_)

print(cm)
print(cm.get_all_metrics())
