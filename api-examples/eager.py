
from baseline.utils import listify
import baseline
from baseline.tf.layers import *
from baseline.tf.embeddings import LookupTableEmbeddings
from baseline.w2v import PretrainedEmbeddingsModel, RandomInitVecModel
import tensorflow as tf
import logging
import numpy as np
tf.enable_eager_execution()

### LIB



### END LIB

NUM_PREFETCH = 2
SHUF_BUF_SZ = 5000
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


def to_tensors(ts):
    X = []
    y = []
    for sample in ts:
        X.append(sample['word'].squeeze())
        y.append(sample['y'].squeeze())
    return np.stack(X), np.stack(y)


W2V_MODEL = '/home/dpressel/.bl-data/281bc75825fa6474e95a1de715f49a3b4e153822'
TS = '/home/dpressel/dev/work/baseline/data/stsa.binary.phrases.train'
VS = '/home/dpressel/dev/work/baseline/data/stsa.binary.dev'
ES = '/home/dpressel/dev/work/baseline/data/stsa.binary.test'

feature_desc = {
    'word': {
        'vectorizer': baseline.Token1DVectorizer(mxlen=100, transform_fn=baseline.lowercase),
        'embed': {'file': W2V_MODEL, 'type': 'default', 'unif': 0.25}
    }
}
# Create a reader that is using our vectorizers to parse a TSV file
# with rows like:
# <label>\t<sentence>\n

vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}
reader = baseline.TSVSeqLabelReader(vectorizers, clean_fn=baseline.TSVSeqLabelReader.do_clean)

train_file = TS
valid_file = VS
test_file = ES

# This builds a set of counters
vocabs, labels = reader.build_vocab([train_file,
                                     valid_file,
                                     test_file])

# This builds a set of embeddings objects, these are typically not DL-specific
# but if they happen to be addons, they can be
embeddings = dict()
for k, v in feature_desc.items():
    embed_config = v['embed']
    embeddings_for_k = baseline.load_embeddings('word', embed_file=embed_config['file'], known_vocab=vocabs[k],
                                                embed_type=embed_config.get('type', 'default'),
                                                unif=embed_config.get('unif', 0.), use_mmap=True)

    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']


X_train, y_train = to_tensors(reader.load(train_file, vocabs=vocabs, batchsz=1))
X_valid, y_valid = to_tensors(reader.load(valid_file, vocabs=vocabs, batchsz=1))
X_test, y_test = to_tensors(reader.load(test_file, vocabs=vocabs, batchsz=1))


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    dataset = dataset.batch(50)
    #dataset = dataset.map(lambda x, y: (x, y))
    dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': tf.count_nonzero(x, axis=1)}, y))
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(NUM_PREFETCH)
    _ = dataset.make_one_shot_iterator()
    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    dataset = dataset.batch(50)
    dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': tf.count_nonzero(x, axis=1)}, y))
    #dataset = dataset.map(lambda x, y: (x, y))
    _ = dataset.make_one_shot_iterator()
    return dataset


def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': tf.count_nonzero(x, axis=1)}, y))
    _ = dataset.make_one_shot_iterator()
    return dataset


model = EmbedPoolStackModel(2, embeddings, ParallelConv(300, 100, [3, 4, 5]), Highway(300))

train_loss_results = []
train_accuracy_results = []

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

global_step = tf.Variable(0)


def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)



def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

import time
num_epochs = 2
for epoch in range(num_epochs):


    # Training loop - using batches of 32
    loss_acc = 0.
    step = 0
    start = time.time()
    for x, y in train_input_fn():
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step)

        loss_acc += loss_value
        step += 1
    print('training time {}'.format(time.time() - start))

    print(model.summary())
    mean_loss = loss_acc / step
    print('Training Loss {}'.format(mean_loss))

    cm = baseline.ConfusionMatrix(['0', '1'])
    for x, y in eval_input_fn():
        # Optimize the model

        # Track progress
        # compare predicted label to actual label
        y_ = tf.argmax(model(x), axis=1, output_type=tf.int32)
        cm.add_batch(y, y_)

    print(cm)
    print(cm.get_all_metrics())

print('FINAL')
cm = baseline.ConfusionMatrix(['0', '1'])
for x, y in predict_input_fn():
    # Optimize the model

    # Track progress
    # compare predicted label to actual label
    y_ = tf.argmax(model(x), axis=1, output_type=tf.int32)
    cm.add_batch(y, y_)

print(cm)
print(cm.get_all_metrics())