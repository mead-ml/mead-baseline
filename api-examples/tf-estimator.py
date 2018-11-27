import baseline as bl
import baseline.tf.embeddings
import baseline.tf.classify
import tensorflow as tf
import numpy as np
import os
import argparse
tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='Train a Baseline model with TensorFlow Estimator API')
parser.add_argument('--export_dir', help='Directory for TF export (for serving)', default='./models', type=str)
parser.add_argument('--checkpoint_dir', help='Directory for model checkpoints', default='./checkpoints', type=str)
parser.add_argument('--model_type', help='What type of model to build', type=str, default='default')
parser.add_argument('--poolsz', help='How many hidden units for pooling', type=int, default=200)
parser.add_argument('--stacksz', help='How many hidden units for stacking', type=int, nargs='+')
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--backend', help='backend', default='tf')
parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) signature name', type=str)
parser.add_argument('--epochs', help='Number of epochs to train', type=int, default=2)
parser.add_argument('--batchsz', help='Batch size', type=int, default=50)
parser.add_argument('--gpus', help='Num GPUs', type=int, default=1)
parser.add_argument('--filts', help='Parallel convolution filter widths (if default model)', type=int, default=[3, 4, 5], nargs='+')
parser.add_argument('--mxlen', help='Maximum post length (number of words) during training', type=int, default=40)
parser.add_argument('--train', help='Training file', default='../data/stsa.binary.phrases.train')
parser.add_argument('--valid', help='Validation file', default='../data/stsa.binary.dev')
parser.add_argument('--test', help='Testing file', default='../data/stsa.binary.test')
parser.add_argument('--embeddings', help='Pretrained embeddings file', default='/data/embeddings/GoogleNews-vectors-negative300.bin')

args = parser.parse_known_args()[0]

pool_field = 'cmotsz' if args.model_type == 'default' else 'rnnsz'

model_params = {
    'model_type': args.model_type,
    'filtsz': args.filts,
    pool_field: args.poolsz
}

if args.stacksz is not None:
    model_params['hsz'] = args.stacksz




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
        'vectorizer': bl.Token1DVectorizer(mxlen=args.mxlen),
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


X_train, y_train = to_tensors(reader.load(train_file, vocabs=vocabs, batchsz=1))
X_valid, y_valid = to_tensors(reader.load(valid_file, vocabs=vocabs, batchsz=1))
X_test, y_test = to_tensors(reader.load(test_file, vocabs=vocabs, batchsz=1))


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(X_train))
    dataset = dataset.batch(args.batchsz)
    dataset = dataset.map(lambda x, y: ({'word': x}, y))
    dataset = dataset.repeat()
    _ = dataset.make_one_shot_iterator()
    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    dataset = dataset.batch(args.batchsz)
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
    return len(X_train)//args.batchsz


def fit(estimator):

    for i in range(args.epochs):

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
        model = bl.model.create_model(embeddings, labels=params['labels'], word=features['word'], y=None, **model_params)
        predictions = {
            'classes': model.best,
            'probabilities': model.probs,
            'logits': model.logits,
        }
        outputs = tf.estimator.export.PredictOutput(predictions['classes'])
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs={'classes': outputs})

    elif mode == tf.estimator.ModeKeys.EVAL:
        bl.tf.SET_TRAIN_FLAG(False)
        model = bl.model.create_model(embeddings, labels=params['labels'], word=features['word'], y=y, **model_params)
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
    model = bl.model.create_model(embeddings, labels=params['labels'], word=features['word'], y=y, **model_params)

    optimizer = tf.train.AdamOptimizer()
    loss = model.create_loss()
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, predictions=model.logits,
                                      loss=loss,
                                      train_op=train_op)
params = {'labels': labels}

if args.gpus > 1:
    distribute = tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpus)
else:
    distribute = None

# Pass to RunConfig
checkpoint_dir = '{}-{}'.format(args.checkpoint_dir, os.getpid())
config = tf.estimator.RunConfig(model_dir=checkpoint_dir)
estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)

fit(estimator)

export_dir = '{}-{}'.format(args.export_dir, os.getpid())
estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=server_input_fn)

