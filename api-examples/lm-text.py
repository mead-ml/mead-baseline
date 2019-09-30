import baseline as bl
import argparse
import os
parser = argparse.ArgumentParser(description='Generate subsequent text by repeatedly predicting the next word from a '
                                             'language model')
parser.add_argument('--model', help='A language model', required=True, type=str)
parser.add_argument('--text', help='raw value, a string', type=str)
parser.add_argument('--device', help='device')


args = parser.parse_known_args()[0]

if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split()
            texts += text  # consider the whole file as a long sequence input

else:
    texts = args.text.split()

print(texts)

m = bl.LanguageModelService.load(args.model, device=args.device)
print(m.predict(texts))
