import baseline as bl
import baseline.tf.classify as classify
import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)
import os
EPOCHS = 2
CMOTSZ = 200
BATCHSZ = 50
NUM_GPUS = 1
FILTS = [3, 4, 5]
BP = '../data'
TRAIN = 'stsa.binary.phrases.train'.format(BP)
VALID = 'stsa.binary.dev'
TEST = 'stsa.binary.test'
W2V_GN_300 = '/data/embeddings/GoogleNews-vectors-negative300.bin'
# The `vectorizer`'s job is to take in a set of tokens and turn them into a numpy array

def to_tensors(ts):
    X = []
    y = []
    for sample in ts:
        X.append(sample['word'].squeeze())
        y.append(sample['y'].squeeze())
    return np.stack(X), np.stack(y)

feature_desc = {
    'word': {
        'vectorizer': bl.Token1DVectorizer(mxlen=40),
        'embed': {'file': W2V_GN_300, 'type': 'default', 'unif': 0.25 }
    }
}
# Create a reader that is using our vectorizers to parse a TSV file
# with rows like:
# <label>\t<sentence>\n

vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}
reader = bl.TSVSeqLabelReader(vectorizers,
                              clean_fn=bl.TSVSeqLabelReader.do_clean)

train_file = os.path.join(BP, TRAIN)
valid_file = os.path.join(BP, VALID)
test_file = os.path.join(BP, TEST)

# This builds a set of counters
vocabs, labels = reader.build_vocab([train_file,
                                     valid_file,
                                     test_file])

# This builds a set of embeddings objects, these are typically not DL-specific
# but if they happen to be addons, they can be
embeddings = dict()
for k, v in feature_desc.items():
    embed_config = v['embed']
    embeddings_for_k = bl.load_embeddings('word', embed_file=embed_config['file'], known_vocab=vocabs[k], embed_type=embed_config.get('type', 'default'), unif=embed_config.get('unif', 0.), use_mmap=True)

    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']


X_train, y_train = to_tensors(reader.load(train_file, vocabs=vocabs, batchsz=1))
X_valid, y_valid = to_tensors(reader.load(valid_file, vocabs=vocabs, batchsz=1))
X_test, y_test = to_tensors(reader.load(test_file, vocabs=vocabs, batchsz=1))


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(X_train))
    dataset = dataset.batch(BATCHSZ)
    dataset = dataset.map(lambda x, y: ({'word': x}, y))
    dataset = dataset.repeat()
    _ = dataset.make_one_shot_iterator()
    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    dataset = dataset.batch(BATCHSZ)
    dataset = dataset.map(lambda x, y: ({'word': x}, y))
    _ = dataset.make_one_shot_iterator()
    return dataset


def server_input_fn():
    tensors = {
        'word': tf.placeholder(tf.int64, [None, None])
    }
    features = {
        'word': tensors['word']
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=tensors, features=features)


def one_epoch(X_train):
    return len(X_train)//BATCHSZ


def fit(estimator):

    for i in range(EPOCHS):

        estimator.train(input_fn=train_input_fn, steps=one_epoch(X_train))

        eval_results = estimator.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    predictions = np.array([p['classes'] for p in estimator.predict(input_fn=eval_input_fn)])
    print(predictions)


def model_fn(features, labels, mode, params):

    if labels is not None:
        y = tf.one_hot(tf.reshape(labels, [-1, 1]), 2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        bl.tf.SET_TRAIN_FLAG(False)
        model = bl.model.create_model(embeddings, labels=params['labels'], word=features['word'], y=None, cmotsz=CMOTSZ, filtsz=FILTS)
        predictions = {
            'classes': model.best,
            'probabilities': model.probs,
            'logits': model.logits,
        }
        outputs = tf.estimator.export.PredictOutput(predictions['classes'])
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs={'classes': outputs})

    elif mode == tf.estimator.ModeKeys.EVAL:
        bl.tf.SET_TRAIN_FLAG(False)
        model = bl.model.create_model(embeddings, labels=params['labels'], word=features['word'], y=y, cmotsz=CMOTSZ, filtsz=FILTS)
        loss = model.create_loss()
        predictions = {
            'classes': model.best,
            'probabilities': model.probs,
            'logits': model.logits,
        }
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels, predictions=predictions['classes'])}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.logits, loss=loss, eval_metric_ops=eval_metric_ops)

    bl.tf.SET_TRAIN_FLAG(True)
    model = bl.model.create_model(embeddings, labels=params['labels'], word=features['word'], y=y, cmotsz=CMOTSZ, filtsz=FILTS)

    optimizer = tf.train.AdamOptimizer()
    loss = model.create_loss()
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, predictions=model.logits,
                                      loss=loss,
                                      train_op=train_op)
params = {'labels': labels}

if NUM_GPUS > 1:
    distribute = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
else:
    distribute = None

# Pass to RunConfig
config = tf.estimator.RunConfig(model_dir='./sst2-{}'.format(os.getpid()))
estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)

fit(estimator)
estimator.export_savedmodel(export_dir_base='./models/sst2', serving_input_receiver_fn=server_input_fn)

