import baseline as bl
import baseline.tf.tagger as tagger
import argparse
import os

parser = argparse.ArgumentParser(description='Tag text with a model')
parser.add_argument('--model', help='A tagger model with extended features', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)


args = parser.parse_known_args()[0]

# This has a special feature named `sidecar` that will be used for a second embedding
vecs = { 
    'word': bl.Dict1DVectorizer(), 
    'sidecar': bl.Dict1DVectorizer(),
    'char': bl.Dict2DVectorizer()
} 

if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split()
            texts += [text]

else:
    texts = [args.text.split()]

print(texts)
m = tagger.load_model(args.model)
print(m.predict_text(texts, vectorizers=vecs))
