import baseline
import baseline.tf.embeddings
import eight_mile.tf.layers as L
import tensorflow as tf
import logging
import numpy as np
import time

# This doesnt produce expected results and appears to be due to a severe bug in tf Keras in handling "deferred" modules
# (ones where the input shape is not declared)
#
# https://github.com/tensorflow/tensorflow/issues/25175#issuecomment-457755125
# The solution right now appears to be not to use keras fit API -- use eager or MEAD instead
# we could change 8mi to declare all this, but need to investigate what the API effect would be
NC = 2
NUM_PREFETCH = 2
#SHUF_BUF_SZ = 500

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
        yc = sample['y'].squeeze() ##.astype(np.float32)
        y.append(yc)
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


def one_epoch(s, batchsz=50):
    """

    effective_batch_sz = args.batchsz = gpu_batchsz*args.gpus

    :param X_train:
    :return:
    """
    return len(s)//batchsz


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(X_train))
    dataset = dataset.batch(50)
    dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': tf.count_nonzero(x, axis=-1)}, y))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(NUM_PREFETCH)
    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    dataset = dataset.batch(50)
    dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': tf.count_nonzero(x, axis=-1)}, y))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(NUM_PREFETCH)

    return dataset


def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda x, y: ({'word': x, 'lengths': tf.count_nonzero(x, axis=-1)}, y))

    return dataset


model = L.EmbedPoolStackModel(NC, embeddings, L.ParallelConv(300, 100, [3, 4, 5])) #, L.Highway(300))


# The compile step specifies the training configuration
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Trains for 5 epochs.
model.fit(train_input_fn().make_one_shot_iterator(),
          epochs=2,
          steps_per_epoch=one_epoch(X_train),
          validation_data=eval_input_fn().make_one_shot_iterator(),
          validation_steps=one_epoch(X_valid),
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_sparse_categorical_accuracy')],)
print(model.summary())
print(model.evaluate(predict_input_fn().make_one_shot_iterator(), steps=len(y_test)))