import baseline
from baseline.pytorch.optz import OptimizerManager
import baseline.pytorch.embeddings
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


train = reader.load(train_file, vocabs=vocabs, batchsz=20)
valid = reader.load(valid_file, vocabs=vocabs, batchsz=20)
test = reader.load(test_file, vocabs=vocabs, batchsz=20)

model = L.EmbedPoolStackModel(2, embeddings, L.ParallelConv(300, 100, [3, 4, 5]), L.Highway(300)).cuda()

train_loss_results = []
train_accuracy_results = []

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = OptimizerManager(model, optim="adam", lr=0.001)


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
        optimizer.zero_grad()
        l = loss(model, x, y)
        l.backward()
        optimizer.step()

        loss_acc += float(l)
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
