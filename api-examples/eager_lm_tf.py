import baseline
import argparse
import baseline.embeddings
import baseline.tf.embeddings
import eight_mile.tf.layers as L
from eight_mile.utils import Timer, revlut, str2bool
from eight_mile.tf.layers import SET_TRAIN_FLAG, set_tf_log_level
from eight_mile.tf.optz import EagerOptimizer
import tensorflow as tf
import logging
import numpy as np

NUM_PREFETCH = 2
SHUF_BUF_SZ = 5000


class Data:

    def __init__(self, ts, batchsz):
        self.x, self.y = Data._to_tensors(ts)


def to_tensors(ts):
    X = []
    y = []
    for sample in ts:
        X.append(sample['word'].squeeze())
        y.append(sample['y'].squeeze())
    return np.stack(X), np.stack(y)


parser = argparse.ArgumentParser(description='Train a Layers model with TensorFlow API')
parser.add_argument('--model_type', help='What type of model to build', type=str, default='default')
parser.add_argument('--hsz', help='How many hidden units for pooling', type=int, default=200)
parser.add_argument('--layers', help='How many layers', type=int, default=2)
parser.add_argument('--stacksz', help='How many hidden units for stacking', type=int, nargs='+')
parser.add_argument('--name', help='(optional) signature name', type=str)
parser.add_argument('--epochs', help='Number of epochs to train', type=int, default=2)
parser.add_argument('--batchsz', help='Batch size', type=int, default=20)
parser.add_argument('--nctx', help='Context steps', type=int, default=35)
parser.add_argument('--train', help='Training file', default='../data/ptb/train.txt')
parser.add_argument('--valid', help='Validation file', default='../data/ptb/valid.txt')
parser.add_argument('--test', help='Testing file', default='../data/ptb/test.txt')
parser.add_argument('--embeddings', help='Pretrained embeddings file', default='/data/embeddings/GoogleNews-vectors-negative300.bin')
parser.add_argument('--ll', help='Log level', type=str, default='info')
parser.add_argument('--lr', help='Learning rate', type=float, default=0.02)
parser.add_argument('--temperature', help='Sample temperature during generation', default=1.0)
parser.add_argument('--start_word', help='Sample start word', default='the')
parser.add_argument('--dropout', help='Dropout', type=float, default=0.1)
parser.add_argument('--num_heads', help='Number of heads (only for Transformer)', type=int, default=4)
parser.add_argument('--transformer', help='Are we using a Transformer (default is LSTM) LM', type=str2bool, default=False)
args = parser.parse_known_args()[0]

embed_type = 'learned-positional' if args.transformer else 'default'
feature_desc = {
    'word': {
        'vectorizer': baseline.Token1DVectorizer(mxlen=-1, transform_fn=baseline.lowercase),
        'embed': {'embed_file': args.embeddings, 'embed_type': embed_type, 'unif': 0.05}
    }
}

set_tf_log_level('ERROR')
vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}
reader = baseline.LineSeqReader(vectorizers, nctx=args.nctx)

train_file = args.train
valid_file = args.valid
test_file = args.test


# This builds a set of counters
vocabs = reader.build_vocab([train_file,
                             valid_file,
                             test_file])



# This builds a set of embeddings objects, these are typically not DL-specific
# but if they happen to be addons, they can be
embeddings = dict()
for k, v in feature_desc.items():
    embed_config = v['embed']
    embeddings_for_k = baseline.embeddings.load_embeddings(k, known_vocab=vocabs[k], **embed_config)
    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']


X_train, y_train = to_tensors(reader.load(train_file, vocabs=vocabs, batchsz=1, tgt_key="word"))
X_valid, y_valid = to_tensors(reader.load(valid_file, vocabs=vocabs, batchsz=1, tgt_key="word"))
X_test, y_test = to_tensors(reader.load(test_file, vocabs=vocabs, batchsz=1, tgt_key="word"))


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    dataset = dataset.batch(args.batchsz, True)
    dataset = dataset.map(lambda x, y: ({'word': x}, y))
    dataset = dataset.prefetch(NUM_PREFETCH)
    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    dataset = dataset.batch(args.batchsz, True)
    dataset = dataset.map(lambda x, y: ({'word': x}, y))
    return dataset


def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda x, y: ({'word': x}, y))
    return dataset


if args.transformer:
    transducer = L.TransformerEncoderStackWithTimeMask(args.num_heads, d_model=args.hsz, layers=args.layers, pdrop=args.dropout)
else:
    transducer = L.LSTMEncoderWithState(None, args.hsz, args.layers, pdrop=args.dropout)
model = L.LangSequenceModel(embeddings["word"].get_vsz(), L.EmbeddingsStack(embeddings), transducer)


def generate_text(model, start_string, temperature=1.0, num_generate=20):
    input_eval = np.array([vocabs["word"].get(s) for s in start_string.split()], dtype=np.int32).reshape(1, -1)
    rlut = revlut(vocabs["word"])
    # Empty string to store our results
    text_generated = [start_string]

    h = None
    for i in range(num_generate):
        predictions, h = model({"word": input_eval, "h": h})
        # remove the batch dimension
        predictions = tf.nn.softmax(predictions / temperature, axis=-1)
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(rlut[predicted_id])

    return text_generated


def loss(model, h, x, y):
    x["h"] = h
    logits, h = model(x)
    vsz = embeddings["word"].get_vsz()
    targets = tf.reshape(y, [-1])
    bt_x_v = tf.nn.log_softmax(tf.reshape(logits, [-1, vsz]), axis=-1)
    one_hots = tf.one_hot(targets, vsz)
    example_loss = -tf.reduce_sum(one_hots * bt_x_v, axis=-1)
    loss = tf.reduce_mean(example_loss)
    return loss, h


optimizer = EagerOptimizer(loss, optim="adam", lr=args.lr)
for epoch in range(args.epochs):


    loss_accum = 0.
    step = 0
    timer.start()
    h = None


    SET_TRAIN_FLAG(True)

    for x, y in train_input_fn():
        # Optimize the model
        loss_value, h = optimizer.update_with_hidden(model, h, x, y)
        loss_accum += loss_value
        step += 1
    print('training time {}'.format(timer.elapsed()))

    mean_loss = loss_accum / step
    print('Training Loss {}, Perplexity {}'.format(mean_loss, np.exp(mean_loss)))


    step = 0
    loss_accum = 0
    SET_TRAIN_FLAG(False)

    for x, y in eval_input_fn():
        # Track progress
        # compare predicted label to actual label
        loss_value, h = loss(model, h, x, y)
        loss_accum += loss_value
        step += 1

    mean_loss = loss_accum / step
    print('Valid Loss {}, Perplexity {}'.format(mean_loss, np.exp(mean_loss)))

    text = generate_text(model, args.start_word, args.temperature)
    print(' '.join(text))
