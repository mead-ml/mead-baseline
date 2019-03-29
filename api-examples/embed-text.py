import os
import argparse
import numpy as np
import baseline as bl

parser = argparse.ArgumentParser(description='Embed text and save to a .npy')
parser.add_argument('--model', help='An embedding model', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--backend', help='backend', default='tf')
parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) service name', type=str)
parser.add_argument('--device', help='device')
parser.add_argument('--preproc', help='(optional) where to perform preprocessing', choices={'client', 'server'}, default='client')
args = parser.parse_args()

if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split()
            texts += [text]
    out = os.path.splitext(args.text)[0]
else:
    texts = [args.text.split()]
    out = 'cli_text'

m = bl.EmbeddingsService.load(
    args.model, backend=args.backend,
    remote=args.remote, name=args.name,
    preproc=args.preproc, device=args.device,
)

embedded = m.predict(texts, preproc=args.preproc)

np.save(out, embedded)
