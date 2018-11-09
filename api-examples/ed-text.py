from __future__ import print_function
import baseline as bl
import argparse
import os
parser = argparse.ArgumentParser(description='Classify text with a model')
parser.add_argument('--model', help='A classifier model', required=True, type=str)
parser.add_argument('--text', help='raw value or a file', type=str)
parser.add_argument('--backend', help='backend', default='tf')
parser.add_argument('--target', help='A file to write decoded output (or print to screen)')
parser.add_argument('--tsv', help='print tab separated', type=bl.str2bool, default=False)
parser.add_argument('--batchsz', help='Size of a batch to pass at once', default=256)
args = parser.parse_known_args()[0]

batches = []
if os.path.exists(args.text) and os.path.isfile(args.text):
    with open(args.text, 'r') as f:
        batch = []
        for line in f:
            text = line.strip().split()
            if len(batch) == args.batchsz:
                batches.append(batch)
                batch = []
            batch.append(text)

        if len(batch) > 0:
            batches.append(batch)

else:
    batch = [args.text.split()]
    batches.append(batch)

m = bl.EncoderDecoderService.load(args.model, backend=args.backend)

f = open(args.target, 'w') if args.target is not None else None

for texts in batches:
    decoded = m.predict(texts)
    for src, dst in zip(texts, decoded):
        src_str = ' '.join(src)
        dst_str = ' '.join(dst)
        if args.tsv:
            line = src_str + '\t' + dst_str
        else:
            line = dst_str

        print(line, file=f)

if f is not None:
    f.close()