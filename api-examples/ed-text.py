import baseline as bl
import argparse
import os
parser = argparse.ArgumentParser(description='Classify text with a model')
parser.add_argument('--model', help='A classifier model', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--backend', help='backend', default='tf')


args = parser.parse_known_args()[0]

if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split()
            texts += [text]

else:
    texts = [args.text.split()]

print(texts)

m = bl.EncoderDecoderService.load(args.model, backend=args.backend)
print(m.predict(texts))
