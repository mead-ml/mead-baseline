import argparse
import baseline.embeddings
from eight_mile.confusion import ConfusionMatrix
import baseline
from eight_mile.pytorch.optz import OptimizerManager, EagerOptimizer
import baseline.pytorch.embeddings
import eight_mile.pytorch.layers as L
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import logging
import numpy as np
import time
import torch


def to_device(m):
    return m.cuda()


def to_host(o):
    return o.cpu().float().numpy()


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
parser.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device (cuda or cpu)")
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

class DictionaryDatasetWrapper(Dataset):
    def __init__(self, x, x_lengths, y):
        self.tensor_dataset = TensorDataset(x, x_lengths, y)

    def __getitem__(self, index):
        # stuff
        x, x_length, y = self.tensor_dataset[index]
        return {'word': x.to(args.device), "lengths": x_length.to(args.device)}, y.to(args.device)

    def __len__(self):
        return len(self.tensor_dataset)


class Data:

    def __init__(self, ts, batchsz):
        self.ds = self._to_tensors(ts)
        self.batchsz = batchsz

    def _to_tensors(self, ts):
        x = []
        x_lengths = []
        y = []
        for sample in ts:
            x.append(sample['word'].squeeze())
            x_lengths.append(sample['word_lengths'].squeeze())
            y.append(sample['y'].squeeze())
        return DictionaryDatasetWrapper(torch.tensor(np.stack(x), dtype=torch.long), torch.tensor(np.stack(x_lengths), dtype=torch.long), torch.tensor(np.stack(y), dtype=torch.long))

    def get_input(self, training=False):
        return DataLoader(self.ds, batch_size=self.batchsz, shuffle=training)


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
    embeddings_for_k = baseline.embeddings.load_embeddings('word', embed_file=embed_config['file'], known_vocab=vocabs[k],
                                                           embed_type=embed_config.get('type', 'default'),
                                                           unif=embed_config.get('unif', 0.), use_mmap=True)

    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']


train_set = Data(reader.load(train_file, vocabs=vocabs, batchsz=1), args.batchsz)
valid_set = Data(reader.load(valid_file, vocabs=vocabs, batchsz=1), args.batchsz)
test_set = Data(reader.load(test_file, vocabs=vocabs, batchsz=1), args.batchsz)

stacksz = len(args.filts) * args.poolsz
num_epochs = 2

model = to_device(
    L.EmbedPoolStackModel(2, L.EmbeddingsStack(embeddings), L.WithoutLength(L.ParallelConv(300, args.poolsz, args.filts)), L.Highway(stacksz))
)


def loss(model, x, y):
    y_ = model(x)
    l = F.nll_loss(y_, y)
    return l


optimizer = EagerOptimizer(loss, optim="adam", lr=0.001)

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
        with torch.no_grad():
            y_ = np.argmax(to_host(model(x)), axis=1)
            cm.add_batch(y, y_)
    print(cm)
    print(cm.get_all_metrics())

print('FINAL')
cm = ConfusionMatrix(['0', '1'])
with torch.no_grad():
    for x, y in test_set.get_input():
        y_ = np.argmax(to_host(model(x)), axis=1)
        cm.add_batch(y, y_)

print(cm)
print(cm.get_all_metrics())
