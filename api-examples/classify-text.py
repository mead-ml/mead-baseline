import baseline as bl
import argparse
import os
from baseline.utils import str2bool

parser = argparse.ArgumentParser(description='Classify text with a model')
parser.add_argument('--model', help='A directory containing classifier model files', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--backend', help='backend', default='tf')
parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) service name', type=str)
parser.add_argument('--preproc', help='(optional) set to true if want to use preproc', type=str2bool, default=False)

args = parser.parse_known_args()[0]

if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split(" ")
            texts += [text]

else:
    texts = [args.text.split(" ")]

m = bl.ClassifierService.load(args.model, backend=args.backend, remote=args.remote, name=args.name, preproc=args.preproc)
for text, output in zip(texts, m.predict(texts, preproc=args.preproc)):
    print("{},{}".format(" ".join(text), output[0][0].decode('ascii')))
