
from eight_mile.utils import listify
import baseline
import eight_mile.tf.layers as L
from eight_mile.utils import get_version
import eight_mile.embeddings
from eight_mile.tf.embeddings import LookupTableEmbeddings
from eight_mile.w2v import PretrainedEmbeddingsModel, RandomInitVecModel
from eight_mile.tf.optz import EagerOptimizer
from baseline.tf.tfy import SET_TRAIN_FLAG
import tensorflow as tf
import logging
import numpy as np


TF_VERSION = get_version(tf)
if TF_VERSION < 2:
    from tensorflow import count_nonzero
    tf.enable_eager_execution()
    Optimizer = tf.train.GradientDescentOptimizer

else:
    from tensorflow.compat.v1 import count_nonzero
    Optimizer = tf.optimizers.SGD

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
    embeddings_for_k = eight_mile.embeddings.load_embeddings(k, known_vocab=vocabs[k], **embed_config)
    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']


X_train, Xch_train, y_train = to_tensors(reader.load(train_file, vocabs=vocabs, batchsz=1)[0])
X_valid, Xch_valid, y_valid = to_tensors(reader.load(valid_file, vocabs=vocabs, batchsz=1)[0])
X_test, Xch_test, y_test = to_tensors(reader.load(test_file, vocabs=vocabs, batchsz=1)[0])


def train_input_fn():
    SET_TRAIN_FLAG(True)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Xch_train, y_train))
    dataset = dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    dataset = dataset.batch(20)
    #dataset = dataset.map(lambda x, y: (x, y))
    dataset = dataset.map(lambda x, xch, y: ({'word': x, 'char': xch, 'lengths': count_nonzero(x, axis=1)}, y))
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(NUM_PREFETCH)
    #_ = dataset.make_one_shot_iterator()
    return dataset


def eval_input_fn():
    SET_TRAIN_FLAG(False)
    dataset = tf.data.Dataset.from_tensor_slices((X_valid, Xch_valid, y_valid))
    dataset = dataset.batch(20)
    dataset = dataset.map(lambda x, xch, y: ({'word': x, 'char': xch, 'lengths': count_nonzero(x, axis=1)}, y))
    #dataset = dataset.map(lambda x, y: (x, y))
    #_ = dataset.make_one_shot_iterator()
    return dataset


def predict_input_fn():
    SET_TRAIN_FLAG(False)
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda x, xch, y: ({'word': x, 'char': xch, 'lengths': count_nonzero(x, axis=1)}, y))
    return dataset


transducer = L.BiLSTMEncoder(None, 200, 2, 0.5, output_fn=L.rnn_signal)
model = L.TagSequenceModel(len(labels), embeddings, transducer)

train_loss_results = []
train_accuracy_results = []


def loss(model, x, y):
  unary = model.transduce(x)
  return model.decoder_model.neg_log_loss(unary, y, x['lengths'])


optim = EagerOptimizer(loss, Optimizer(learning_rate=0.01))


import time
num_epochs = 20
for epoch in range(num_epochs):


    # Training loop - using batches of 32
    loss_acc = 0.
    step = 0
    start = time.time()
    for x, y in train_input_fn():
        # Optimize the model
        loss_value = optim.update(model, x, y)

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

