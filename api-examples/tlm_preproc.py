import argparse
import baseline

from baseline.vectorizers import BPEVectorizer1D
import numpy as np
import json

def mlm_masking(inputs, mask_value, vocab_size):
    labels = np.copy(inputs)
    masked_indices = np.random.binomial(size=len(inputs), n=1, p=0.15)
    masked_indices[np.random.randint(1, len(inputs)-1)] = 1
    # Anything not masked is 0 so no loss
    labels[masked_indices == 0] = 0
    # Of the masked items, mask 80% of them with [MASK]
    indices_replaced = np.random.binomial(size=len(inputs), n=1, p=0.8)
    indices_replaced = indices_replaced & masked_indices
    inputs[indices_replaced == 1] = mask_value
    indices_random = np.random.binomial(size=len(inputs), n=1, p=0.5)
    # Replace 10% of them with random words, rest preserved for auto-encoding
    indices_random = indices_random & masked_indices & ~indices_replaced
    random_words = np.random.randint(low=len(baseline.Offsets.VALUES) + 3, high=vocab_size-1, size=len(inputs))
    inputs[indices_random == 1] = random_words[indices_random == 1]
    return inputs, labels


def create_record(chunk, str_lookup, prefix, suffix, mask_value, vocab_size):
    if prefix:
        chunk = [prefix] + chunk
    if suffix:
        chunk = [suffix] + chunk

    inputs, labels = mlm_masking(np.array(chunk), mask_value, vocab_size)
    return {'x': inputs, 'y': labels, 'x_str': [str_lookup[s] for s in inputs], 'y_str': [str_lookup[s] for s in labels]}


class TSVWriter:
    def __init__(self, name, fields):
        self.fields = fields
        self.writer = open(name, 'w')

        self._write_list(self.fields)

    def _write_list(self, l):
        self.writer.write('\t'.join(l) + '\n')

    def write(self, record):
        l = [' '.join(record[f]) for f in self.fields]
        self._write_list(l)

    def close(self):
        self.writer.close()

class JSONLWriter:

    def __init__(self, name, fields):
        self.fields = fields
        self.writer = open(name, "w")

    def write(self, record):
        r = {}
        for f in self.fields:
            r[f] = record[f]
        output = json.dumps(r) + '\n'
        self.writer.write(output)

    def close(self):
        self.writer.close()

def create_file_writer(fmt, name, fields):
    if fmt == 'tsv':
        return TSVWriter(name, fields)
    if fmt == 'json':
        return JSONLWriter(name, fields)



parser = argparse.ArgumentParser(description='Convert text into MLM fixed width contexts')

parser.add_argument('--text', help='The text to classify as a string, or a path to a file with each line as an example',
                    type=str)
parser.add_argument('--codes', help='BPE codes')
parser.add_argument('--vocab', help='BPE vocab')
parser.add_argument("--nctx", type=int, default=256, help="Max input length")
parser.add_argument("--fmt", type=str, default='json', choices=['json', 'tsv'])
parser.add_argument("--fields", type=str, nargs="+", default=["x_str", "y_str"])
parser.add_argument("--output", type=str, help="Output name")
parser.add_argument("--prefix", type=str, help="Prefix every line with this token")
parser.add_argument("--suffix", type=str, help="Suffix every line with this token")
parser.add_argument("--stride", type=int, help="Tokens to stride before next read, defaults to `nctx`")
parser.add_argument("--eos_on_eol", type=baseline.str2bool, default=True)
parser.add_argument("--cased", type=baseline.str2bool, default=True)
args = parser.parse_args()
if not args.output:
    args.output = f'{args.text}.records.{args.fmt}'

print(args.output)
transform = baseline.lowercase
vectorizer = BPEVectorizer1D(transform_fn=transform, model_file=args.codes, vocab_file=args.vocab, mxlen=1024)

lookup_indices = []
words = []
indices2word = baseline.revlut(vectorizer.vocab)
vocab_size = max(vectorizer.vocab.values()) + 1
nctx = args.nctx
mask_value = vectorizer.vocab['[MASK]']
prefix = suffix = None
if args.prefix:
    nctx -= 1
    prefix = vectorizer.vocab[args.prefix]

if args.suffix:
    nctx -= 1
    suffix = vectorizer.vocab[args.suffix]

fw = create_file_writer(args.fmt, args.output, args.fields)
with open(args.text, encoding='utf-8') as rf:
    for line in rf:
        to_bpe = line.strip().split()
        if args.eos_on_eol:
            to_bpe += ['<EOS>']

        output, available = vectorizer.run(to_bpe, vectorizer.vocab)
        while available > 0:
            if len(lookup_indices) == nctx:
                record = create_record(lookup_indices, indices2word, prefix, suffix, mask_value, vocab_size)
                fw.write(record)
                lookup_indices = []
            needed = nctx - len(lookup_indices)
            if available >= needed:
                lookup_indices += output[:needed].tolist()
                output = output[needed:]
                available -= needed
                record = create_record(lookup_indices, indices2word, prefix, suffix, mask_value, vocab_size)
                fw.write(record)
                lookup_indices = []
            # The amount available is less than what we need, so read the whole thing
            else:
                lookup_indices += output[:available].tolist()
                available = 0
fw.close()