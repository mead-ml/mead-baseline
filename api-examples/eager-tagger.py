
from baseline.utils import listify
import baseline
import baseline.tf.layers as L

from baseline.tf.embeddings import LookupTableEmbeddings
from baseline.w2v import PretrainedEmbeddingsModel, RandomInitVecModel
import tensorflow as tf
import logging
import numpy as np
tf.enable_eager_execution()

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
    Xch = []
    y = []
    for sample in ts:
        X.append(sample['word'].squeeze())
        Xch.append(sample['char'].squeeze())
        y.append(sample['y'].squeeze())
    return np.stack(X), np.stack(Xch), np.stack(y)


VSM_MODEL = '/home/dpressel/.bl-data/dce69c404025a8312c323197347695e81fd529fc/glove.twitter.27B.200d.txt'

TS = '/home/dpressel/dev/work/baseline/data/oct27.train'
VS = '/home/dpressel/dev/work/baseline/data/oct27.dev'
ES = '/home/dpressel/dev/work/baseline/data/oct27.test'


feature_desc = {
    'word': {
        'vectorizer': baseline.Dict1DVectorizer(mxlen=-1, transform_fn=baseline.lowercase),
        'embed': {'embed_file': VSM_MODEL, 'embed_type': 'default', 'unif': 0.25}
    },
    'char': {
        'vectorizer': baseline.Dict2DVectorizer(mxlen=-1, mxwlen=40),
        'embed': {'dsz': 30, 'embed_type': 'char-conv', 'wsz': 30}
    }
}

vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}
reader = baseline.CONLLSeqReader(vectorizers, named_fields={"0": "text", "-1": "y"})

train_file = TS
valid_file = VS
test_file = ES

# This builds a set of counters
vocabs = reader.build_vocab([train_file,
                             valid_file,
                             test_file])

labels = reader.label2index

# This builds a set of embeddings objects, these are typically not DL-specific
# but if they happen to be addons, they can be
embeddings = dict()
for k, v in feature_desc.items():
    embed_config = v['embed']
    embeddings_for_k = baseline.load_embeddings(k, known_vocab=vocabs[k], **embed_config)
    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']


X_train, Xch_train, y_train = to_tensors(reader.load(train_file, vocabs=vocabs, batchsz=1)[0])
X_valid, Xch_valid, y_valid = to_tensors(reader.load(valid_file, vocabs=vocabs, batchsz=1)[0])
X_test, Xch_test, y_test = to_tensors(reader.load(test_file, vocabs=vocabs, batchsz=1)[0])


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Xch_train, y_train))
    dataset = dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    dataset = dataset.batch(20)
    #dataset = dataset.map(lambda x, y: (x, y))
    dataset = dataset.map(lambda x, xch, y: ({'word': x, 'char': xch, 'lengths': tf.count_nonzero(x, axis=1)}, y))
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(NUM_PREFETCH)
    _ = dataset.make_one_shot_iterator()
    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_valid, Xch_valid, y_valid))
    dataset = dataset.batch(20)
    dataset = dataset.map(lambda x, xch, y: ({'word': x, 'char': xch, 'lengths': tf.count_nonzero(x, axis=1)}, y))
    #dataset = dataset.map(lambda x, y: (x, y))
    _ = dataset.make_one_shot_iterator()
    return dataset


def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda x, xch, y: ({'word': x, 'char': xch, 'lengths': tf.count_nonzero(x, axis=1)}, y))
    _ = dataset.make_one_shot_iterator()
    return dataset


transducer = L.BiLSTMEncoder(200, 0.5, 1, output_fn=L.rnn_signal)
model = L.TagSequenceModel(len(labels), embeddings, transducer)

train_loss_results = []
train_accuracy_results = []

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

global_step = tf.Variable(0)



def loss(model, x, y):
  unary = model.transduce(x)
  return model.decoder_model.neg_log_loss(unary, y, x['lengths'])



def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

import time
num_epochs = 20
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

    mean_loss = loss_acc / step
    print('Training Loss {}'.format(mean_loss))

    acc = 0
    total = 0
    for x, y in eval_input_fn():
        # Optimize the model

        # Track progress
        # compare predicted label to actual label
        y_ = model(x).numpy()
        y = y.numpy()
        #y_ = tf.argmax(model(x), axis=-1, output_type=tf.int32)
        lengths = x['lengths'].numpy()

        B = y.shape[0]
        bacc = 0
        for b in range(B):
            bacc += np.sum(y_[b][:lengths[b]] == y[b][:lengths[b]])
        acc += bacc
        btotal = np.sum(lengths)
        total += btotal
        print('batch acc {}/{}'.format(bacc, btotal))


    print('ACC', acc/float(total))

