import argparse
import baseline
import baseline.embeddings
import baseline.pytorch.embeddings
from collections import Counter
from eight_mile.confusion import ConfusionMatrix
from eight_mile.pytorch.optz import EagerOptimizer
import eight_mile.pytorch.layers as L
import nlp as nlp_datasets
import numpy as np
import os
import time
import torch
import torch.nn.functional as F


def to_device(d):
    if isinstance(d, dict):
        return {k: v.cuda() for k, v in d.items()}
    return d.cuda()


def to_host(o):
    return o.cpu().float().numpy()


def create_vocabs(datasets, vectorizers):
    vocabs = {k: Counter() for k in vectorizers.keys()}
    for dataset in datasets:
        for k, v in vectorizers.items():
            vocabs[k] = Counter()
            for example in dataset:
                vocabs[k] += v.count(example['sentence'].split())
    return vocabs


def create_featurizer(vectorizers, vocabs, primary_key='word'):
    def convert_to_features(batch):

        features = {k: [] for k in vectorizers.keys()}

        features['lengths'] = []

        features['y'] = [l for l in batch['label']]

        for i, text in enumerate(batch['sentence']):
            for k, v in vectorizers.items():
                vec, lengths = v.run(text.split(), vocabs[k])
                if k == primary_key:
                    features['lengths'].append(lengths)
                features[k].append(vec.tolist())

        return features
    return convert_to_features


parser = argparse.ArgumentParser(description='Train a Layers model with PyTorch API')
parser.add_argument('--model_type', help='What type of model to build', type=str, default='default')
parser.add_argument('--poolsz', help='How many hidden units for pooling', type=int, default=100)
parser.add_argument('--dsz', help='Embeddings dimension size', type=int, default=300)
parser.add_argument('--stacksz', help='How many hidden units for stacking', type=int, nargs='+')
parser.add_argument('--name', help='(optional) signature name', type=str)
parser.add_argument('--epochs', help='Number of epochs to train', type=int, default=2)
parser.add_argument('--batchsz', help='Batch size', type=int, default=50)
parser.add_argument('--filts', help='Parallel convolution filter widths (if default model)', type=int, default=[3, 4, 5], nargs='+')
parser.add_argument('--mxlen', help='Maximum post length (number of words) during training', type=int, default=100)
parser.add_argument('--dataset', help='HuggingFace Datasets id', default=['glue', 'sst2'], nargs='+')
parser.add_argument('--embeddings', help='Pretrained embeddings file', default='https://www.dropbox.com/s/699kgut7hdb5tg9/GoogleNews-vectors-negative300.bin.gz?dl=1')
parser.add_argument('--ll', help='Log level', type=str, default='info')
parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('--blcache', help='Cache for embeddings', default=os.path.expanduser('~/.bl-data'))
parser.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device (cuda or cpu)")
args = parser.parse_known_args()[0]

embeddings_file = baseline.EmbeddingDownloader(args.embeddings, embedding_dsz=args.dsz, embedding_sha1=None, data_download_cache=args.blcache).download()

feature_desc = {
    'word': {
        'vectorizer': baseline.Token1DVectorizer(mxlen=100, transform_fn=baseline.lowercase),
        'embed': {'file': embeddings_file, 'type': 'default', 'unif': 0.25, 'dsz': args.dsz}
    }
}

vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}

dataset = nlp_datasets.load_dataset(*args.dataset)
vocabs = create_vocabs(dataset.values(), vectorizers)

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


train_set = dataset['train']
valid_set = dataset['validation']
test_set = dataset['test']

convert_to_features = create_featurizer(vectorizers, vocabs)
train_set = train_set.map(convert_to_features, batched=True)
train_set.set_format(type='torch', columns=list(vectorizers.keys()) + ['y', 'lengths'])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchsz)

valid_set = valid_set.map(convert_to_features, batched=True)
valid_set.set_format(type='torch', columns=list(vectorizers.keys()) + ['y', 'lengths'])
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batchsz)

test_set = test_set.map(convert_to_features, batched=True)
test_set.set_format(type='torch', columns=list(vectorizers.keys()) + ['y', 'lengths'])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batchsz)

stacksz = len(args.filts) * args.poolsz
num_epochs = 2

model = to_device(
    L.EmbedPoolStackModel(2, L.EmbeddingsStack(embeddings), L.WithoutLength(L.ParallelConv(args.dsz, args.poolsz, args.filts)), L.Highway(stacksz))
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
    for x in train_loader:
        x = to_device(x)
        y = x.pop('y')
        loss_value = optimizer.update(model, x, y)
        loss_acc += loss_value
        step += 1
    print('training time {}'.format(time.time() - start))
    mean_loss = loss_acc / step
    print('Training Loss {}'.format(mean_loss))
    cm = ConfusionMatrix(['0', '1'])
    for x in valid_loader:
        x = to_device(x)
        with torch.no_grad():
            y = x.pop('y')
            y_ = np.argmax(to_host(model(x)), axis=1)
            cm.add_batch(y, y_)
    print(cm)
    print(cm.get_all_metrics())

print('FINAL')
cm = ConfusionMatrix(['0', '1'])
with torch.no_grad():
    for x in test_loader:
        x = to_device(x)
        y = x.pop('y')
        y_ = np.argmax(to_host(model(x)), axis=1)
        cm.add_batch(y, y_)

print(cm)
print(cm.get_all_metrics())
