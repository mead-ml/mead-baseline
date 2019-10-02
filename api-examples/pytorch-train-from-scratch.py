import argparse
import eight_mile.embeddings
import baseline
from baseline.pytorch.optz import OptimizerManager
import eight_mile.pytorch.embeddings
import eight_mile.pytorch.layers as L
import torch.nn.functional as F
import logging
import numpy as np
import time
import torch


def get_logging_level(ll):
    ll = ll.lower()
    if ll == 'debug':
        return logging.DEBUG
    if ll == 'info':
        return logging.INFO
    return logging.WARNING

parser = argparse.ArgumentParser(description='Train a Layers model with PyTorch API')
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
# Create a reader that is using our vectorizers to parse a TSV file
# with rows like:
# <label>\t<sentence>\n

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


train = reader.load(train_file, vocabs=vocabs, batchsz=args.batchsz)
valid = reader.load(valid_file, vocabs=vocabs, batchsz=args.batchsz)
test = reader.load(test_file, vocabs=vocabs, batchsz=args.batchsz)

stacksz = len(args.filts) * args.poolsz
model = L.EmbedPoolStackModel(2, embeddings, L.ParallelConv(300, args.poolsz, args.filts), L.Highway(stacksz)).cuda()

train_loss_results = []
train_accuracy_results = []



def loss(model, x, y):
    y_ = model(x)
    l = F.nll_loss(y_, y)
    return l


num_epochs = 2

def as_np(cuda_ten):
    return cuda_ten.cpu().float().numpy()

def make_pair(batch_dict, train=False):

    example_dict = dict({})
    for key in feature_desc.keys():
        example_dict[key] = torch.from_numpy(batch_dict[key]).cuda()

    # Allow us to track a length, which is needed for BLSTMs
    if 'word_lengths' in batch_dict:
        example_dict['lengths'] = torch.from_numpy(batch_dict['word_lengths']).cuda()

    y = batch_dict.pop('y')
    if train:
        y = torch.from_numpy(y).cuda()
    return example_dict, y

class EagerOptimizer(object):

    def __init__(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def update(self, model, x, y):
        self.optimizer.zero_grad()
        l = self.loss(model, x, y)
        l.backward()
        self.optimizer.step()
        return float(l)


optimizer = EagerOptimizer(loss, OptimizerManager(model, optim="adam", lr=args.lr))

for epoch in range(num_epochs):

    # Training loop - using batches of 32
    loss_acc = 0.
    epoch_loss = 0.
    epoch_div = 0.

    step = 0
    start = time.time()
    batchsz = 20
    for b in train:
        x, y = make_pair(b, True)
        loss_value = optimizer.update(model, x, y)
        loss_acc += loss_value
        step += 1

    print('training time {}'.format(time.time() - start))

    mean_loss = loss_acc / step
    print('Training Loss {}'.format(mean_loss))

    cm = baseline.ConfusionMatrix(['0', '1'])
    with torch.no_grad():
        for b in valid:
            x, y = make_pair(b)
            y_ = np.argmax(as_np(model(x)), axis=1)
            cm.add_batch(y, y_)

    print(cm)
    print(cm.get_all_metrics())

print('FINAL')
cm = baseline.ConfusionMatrix(['0', '1'])
with torch.no_grad():
    for b in test:
        x, y = make_pair(b)
        y_ = np.argmax(as_np(model(x)), axis=1)

        cm.add_batch(y, y_)

print(cm)
print(cm.get_all_metrics())
