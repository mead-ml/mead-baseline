import baseline as bl
import argparse
import os
parser = argparse.ArgumentParser(description='Classify text with a model')
parser.add_argument('--model', help='A classifier model', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--backend', help='backend', default='tf')
parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) service name', type=str)
parser.add_argument('--model_type', type=str, default='default')
parser.add_argument('--modules', default=[])
args = parser.parse_known_args()[0]


for mod_name in args.modules:
    bl.import_user_module(mod_name)

if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split()
            texts += [text]

else:
    texts = [args.text.split()]

print(texts)

m = bl.ClassifierService.load(args.model, backend=args.backend, remote=args.remote, name=args.name, model_type=args.model_type)
print(m.predict(texts))
