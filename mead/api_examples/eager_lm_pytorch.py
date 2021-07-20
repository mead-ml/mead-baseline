import baseline
import argparse
import baseline.pytorch.embeddings
import eight_mile.pytorch.layers as L
from eight_mile.utils import revlut, str2bool, Timer
from eight_mile.pytorch.layers import SequenceLoss
from eight_mile.pytorch.optz import EagerOptimizer
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import logging
import numpy as np
import time


def to_device(m):
    return m.cuda()


def to_host(o):
    return o.cpu().float().numpy()


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
parser.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device (cuda or cpu)")
args = parser.parse_known_args()[0]

embed_type = 'learned-positional' if args.transformer else 'default'

feature_desc = {
    'word': {
        'vectorizer': baseline.Token1DVectorizer(mxlen=-1, transform_fn=baseline.lowercase),
        'embed': {'embed_file': args.embeddings, 'embed_type': embed_type, 'unif': 0.05}
    }
}


class DictionaryDatasetWrapper(Dataset):
    def __init__(self, x, y):
        self.tensor_dataset = TensorDataset(x, y)

    def __getitem__(self, index):
        # stuff
        x, y = self.tensor_dataset[index]
        return {'word': x.to(args.device)}, y.to(args.device)

    def __len__(self):
        return len(self.tensor_dataset)


class Data:

    def __init__(self, ts, batchsz):
        self.ds = self._to_tensors(ts)
        self.batchsz = batchsz

    def _to_tensors(self, ts):
        x = []
        y = []
        for sample in ts:
            x.append(sample['word'].squeeze())
            y.append(sample['y'].squeeze())
        return DictionaryDatasetWrapper(torch.tensor(np.stack(x), dtype=torch.long), torch.tensor(np.stack(y), dtype=torch.long))

    def get_input(self, training=False):
        return DataLoader(self.ds, batch_size=self.batchsz, shuffle=training)


vectorizers = {k: v['vectorizer'] for k, v in feature_desc.items()}
reader = baseline.LineSeqReader(vectorizers, nctx=args.nctx)

train_file = args.train
valid_file = args.valid
test_file = args.test


# This builds a set of counters
vocabs = reader.build_vocab([train_file, valid_file, test_file])


# This builds a set of embeddings objects, these are typically not DL-specific
# but if they happen to be addons, they can be
embeddings = dict()
for k, v in feature_desc.items():
    embed_config = v['embed']
    embeddings_for_k = baseline.embeddings.load_embeddings(k, known_vocab=vocabs[k], **embed_config)
    embeddings[k] = embeddings_for_k['embeddings']
    # Reset the vocab to the embeddings one
    vocabs[k] = embeddings_for_k['vocab']


train_set = Data(reader.load(train_file, vocabs=vocabs, batchsz=1, tgt_key="word"), args.batchsz)
valid_set = Data(reader.load(valid_file, vocabs=vocabs, batchsz=1, tgt_key="word"), args.batchsz)
test_set = Data(reader.load(test_file, vocabs=vocabs, batchsz=1, tgt_key="word"), args.batchsz)


if args.transformer:
    transducer = L.TransformerEncoderStackWithTimeMask(args.num_heads, d_model=args.hsz, layers=args.layers,
                                                       pdrop=args.dropout, input_sz=embeddings['word'].get_dsz())
else:
    transducer = L.LSTMEncoderWithState(embeddings['word'].get_dsz(), args.hsz, args.layers, pdrop=args.dropout)
model = to_device(L.LangSequenceModel(embeddings["word"].get_vsz(), L.EmbeddingsStack(embeddings), transducer))


def generate_text(model, start_string, temperature=1.0, num_generate=20):
    input_eval = torch.tensor([vocabs["word"].get(s) for s in start_string.split()]).long().view(1, -1).to(args.device)
    rlut = revlut(vocabs["word"])
    # Empty string to store our results
    text_generated = [start_string]

    h = None
    for i in range(num_generate):
        predictions, h = model({"word": input_eval, "h": h})
        # remove the batch dimension
        predictions = torch.softmax(predictions / temperature, dim=-1)
        predictions = predictions.squeeze(0)

        # using a multinomial distribution to predict the word returned by the model
        predicted_id = torch.multinomial(predictions, num_samples=1)[-1, 0]
        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = predicted_id.unsqueeze(0).unsqueeze(0)

        text_generated.append(rlut[predicted_id.cpu().numpy().item()])

    return text_generated


crit = SequenceLoss(LossFn=torch.nn.CrossEntropyLoss)


def loss(model, h, x, y):
    x["h"] = h
    logits, h = model(x)
    vsz = embeddings["word"].vsz
    l = crit(logits, y)

    return l, h


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


optimizer = EagerOptimizer(loss, optim="sgd", lr=args.lr)

timer = Timer()
for epoch in range(args.epochs):
    loss_accum = 0.
    step = 0
    timer.start()
    h = None

    for x, y in train_set.get_input(training=True):
        # Optimize the model
        if h is not None:
            h = repackage_hidden(h)
        loss_value, h = optimizer.update_with_hidden(model, h, x, y)
        loss_accum += loss_value
        step += 1
    print(f'training time {timer.elapsed()}')

    mean_loss = loss_accum / step
    print(f'Training Loss {mean_loss}, Perplexity {np.exp(mean_loss)}')


    step = 0
    loss_accum = 0

    for x, y in valid_set.get_input(training=False):
        with torch.no_grad():
            # Track progress
            # compare predicted label to actual label
            loss_value, h = loss(model, h, x, y)
            h = repackage_hidden(h)

            loss_accum += to_host(loss_value)
            step += 1

    mean_loss = loss_accum / step
    print(f'Valid Loss {mean_loss}, Perplexity {np.exp(mean_loss)}')

    text = generate_text(model, args.start_word, args.temperature)
    print(' '.join(text))

for x, y in test_set.get_input(training=False):
    with torch.no_grad():
        loss_value, h = loss(model, h, x, y)
        h = repackage_hidden(h)
        loss_accum += to_host(loss_value)
        step += 1

mean_loss = loss_accum / step
print(f'Test Loss {mean_loss}, Perplexity {np.exp(mean_loss)}')
