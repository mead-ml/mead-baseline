import baseline
import eight_mile.tf.embeddings
import eight_mile.tf.layers as L
from eight_mile.utils import get_version
from eight_mile.tf.layers import TRAIN_FLAG, SET_TRAIN_FLAG
from eight_mile.tf.optz import EagerOptimizer
from eight_mile.utils import listify, revlut
from eight_mile.tf.embeddings import LookupTableEmbeddings
from eight_mile.w2v import PretrainedEmbeddingsModel, RandomInitVecModel
import tensorflow as tf
import logging
import numpy as np

if get_version(tf) < 2:
    tf.enable_eager_execution()
    SGD = tf.train.GradientDescentOptimizer

else:
    SGD = tf.optimizers.SGD
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

TS = '/home/dpressel/dev/work/baseline/data/ptb/train.txt'
VS = '/home/dpressel/dev/work/baseline/data/ptb/valid.txt'
ES = '/home/dpressel/dev/work/baseline/data/ptb/test.txt'


feature_desc = {
    'word': {
        'vectorizer': baseline.Token1DVectorizer(mxlen=-1, transform_fn=baseline.lowercase),
        'embed': {'embed_file': W2V_MODEL, 'embed_type': 'default', 'unif': 0.05}
    }
}

vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}
reader = baseline.LineSeqReader(vectorizers, nctx=35)

train_file = TS
valid_file = VS
test_file = ES

# This builds a set of counters
vocabs = reader.build_vocab([train_file,
                             valid_file,
                             test_file])



# This builds a set of embeddings objects, these are typically not DL-specific
# but if they happen to be addons, they can be
embeddings = dict()
for k, v in feature_desc.items():
    embed_config = v['embed']
    embeddings_for_k = eight_mile.embeddings.load_embeddings(k, known_vocab=vocabs[k], **embed_config)
    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']


X_train, y_train = to_tensors(reader.load(train_file, vocabs=vocabs, batchsz=1, tgt_key="word"))
X_valid, y_valid = to_tensors(reader.load(valid_file, vocabs=vocabs, batchsz=1, tgt_key="word"))
X_test, y_test = to_tensors(reader.load(test_file, vocabs=vocabs, batchsz=1, tgt_key="word"))


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    dataset = dataset.batch(20, True)
    dataset = dataset.map(lambda x, y: ({'word': x}, y))
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(NUM_PREFETCH)
    return dataset


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    dataset = dataset.batch(20, True)
    dataset = dataset.map(lambda x, y: ({'word': x}, y))
    #dataset = dataset.map(lambda x, y: (x, y))
    return dataset


def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda x, y: ({'word': x}, y))
    return dataset

transducer = L.LSTMEncoderWithState(None, 200, 1, 0.5)
model = L.LangSequenceModel(embeddings["word"].vsz, embeddings, transducer)

train_loss_results = []
train_accuracy_results = []

optimizer = SGD(learning_rate=1)

global_step = tf.Variable(0)


def loss(model, h, x, y):

    x["h"] = h
    logits, h = model(x)
    #print('IN LOSS ', h)
    vsz = embeddings["word"].vsz
    targets = tf.reshape(y, [-1])
    bt_x_v = tf.nn.log_softmax(tf.reshape(logits, [-1, vsz]), axis=-1)
    one_hots = tf.one_hot(targets, vsz)
    example_loss = -tf.reduce_sum(one_hots * bt_x_v, axis=-1)
    loss = tf.reduce_mean(example_loss)
    return loss, h


def grad(model, h, inputs, targets):
  with tf.GradientTape() as tape:
      loss_value, h = loss(model, h, inputs, targets)
  return loss_value, h, tape.gradient(loss_value, model.trainable_variables)

"""
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # You can change the start string to experiment
    start_string = 'ROMEO'

    # Converting our start string to numbers (vectorizing)
    input_eval = vectorizers["word"].run(start_string.split())
    rlut = revlut(embeddings["word"].vocab)

    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(rlut[predicted_id])

    return (start_string + ''.join(text_generated))
"""
import time
num_epochs = 2
for epoch in range(num_epochs):


    loss_accum = 0.
    step = 0
    start = time.time()
    h = None
    for x, y in train_input_fn():
        # Optimize the model
        loss_value, h, grads = grad(model, h, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step)

        loss_accum += loss_value
        step += 1
    print('training time {}'.format(time.time() - start))

    mean_loss = loss_accum / step
    print('Training Loss {}, Perplexity {}'.format(mean_loss, np.exp(mean_loss)))


    step = 0
    loss_accum = 0
    for x, y in eval_input_fn():
        # Optimize the model

        # Track progress
        # compare predicted label to actual label
        loss_value, h = loss(model, h, x, y)
        loss_accum += loss_value
        step += 1

    mean_loss = loss_accum / step
    print('Valid Loss {}, Perplexity {}'.format(mean_loss, np.exp(mean_loss)))


