import baseline as bl
import argparse
import os

parser = argparse.ArgumentParser(description='Classify text with a model')
parser.add_argument('--model', help='The path to either the .zip file created by training or to the client bundle '
                                    'created by exporting', required=True, type=str)
parser.add_argument('--text', help='The text to classify as a string, or a path to a file with each line as an example',
                    type=str)
parser.add_argument('--backend', help='backend', choices={'tf', 'pytorch'}, default='tf')
parser.add_argument('--remote', help='(optional) remote endpoint, normally localhost:8500', type=str) # localhost:8500
parser.add_argument('--name', help='(optional) service name as the server may serves multiple models', type=str)
parser.add_argument('--device', help='device')
parser.add_argument('--preproc', help='(optional) where to perform preprocessing', choices={'client', 'server'},
                    default='client')
parser.add_argument('--batchsz', help='batch size when --text is a file', default=100, type=int)
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
