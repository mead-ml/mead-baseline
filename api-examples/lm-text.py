import baseline as bl
import argparse
import os
from baseline.utils import str2bool, put_addons_in_path
parser = argparse.ArgumentParser(description='Classify text with a model')
parser.add_argument('--model', help='A classifier model', required=True, type=str)
parser.add_argument('--text', help='raw value', type=str)
parser.add_argument('--device', help='device')
parser.add_argument('--backend', help='backend', choices={'tf', 'pytorch'}, default='tf')
parser.add_argument('--prefer_eager', help="If running in TensorFlow, should we prefer eager model", type=str2bool)
parser.add_argument("--addon_path", type=str, default=os.path.expanduser('~/.bl-data/addons'),
                    help="Path or url of the dataset cache")

args = parser.parse_known_args()[0]

put_addons_in_path(args.addon_path)

if args.backend == 'tf':
    from eight_mile.tf.layers import set_tf_eager_mode
    set_tf_eager_mode(args.prefer_eager)

if os.path.exists(args.text) and os.path.isfile(args.text):
    texts = []
    with open(args.text, 'r') as f:
        for line in f:
            text = line.strip().split()
            texts += [text]

else:
    texts = args.text.split()

print(texts)

m = bl.LanguageModelService.load(args.model, device=args.device)
print(m.predict(texts))
