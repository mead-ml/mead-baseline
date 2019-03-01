import baseline as bl
import argparse
import os

parser = argparse.ArgumentParser(description='Tag text with a model')
parser.add_argument('--model', help='A tagger model with extended features', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--backend', help='backend', default='tf')
parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) signature name', type=str)
parser.add_argument('--preproc', help='(optional) where to perform preprocessing', choices={'client', 'server'}, default='client')

bl.import_user_module("vec_text")

args = parser.parse_known_args()[0]

if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split()
            texts += [text]

else:
    texts = [args.text.split()]

m = bl.TaggerService.load(args.model, backend=args.backend, remote=args.remote, name=args.name, preproc=args.preproc)
for sen in m.predict(texts, preproc=args.preproc):
    for word_tag in sen:
        print("{} {}".format(word_tag['text'], word_tag['label']))
    print("\n")
