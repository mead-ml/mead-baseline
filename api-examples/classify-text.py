import baseline as bl
import argparse
import os

parser = argparse.ArgumentParser(description='Classify text with a model')
parser.add_argument('--model', help='A classifier model', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--backend', help='backend', default='tf')
parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) service name', type=str)
parser.add_argument('--device', help='device')
parser.add_argument('--preproc', help='(optional) where to perform preprocessing', choices={'client', 'server'}, default='client')
parser.add_argument('--batchsz', help='batch data', default=100, type=int)
args = parser.parse_args()

if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split()
            texts += [text]

else:
    texts = [args.text.split()]
batched = [texts[i:i + args.batchsz] for i in range(0, len(texts), args.batchsz)]

m = bl.ClassifierService.load(args.model, backend=args.backend, remote=args.remote,
                              name=args.name, preproc=args.preproc,
                              device=args.device)
for texts in batched:
    for text, output in zip(texts, m.predict(texts)):
        print("{}, {}".format(" ".join(text), output[0][0]))
