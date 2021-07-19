import baseline as bl
import argparse
import os
from eight_mile.utils import str2bool

def main():
    parser = argparse.ArgumentParser(description='Classify text with a model')
    parser.add_argument('--model', help='A classifier model', required=True, type=str)
    parser.add_argument('--text', help='raw value', type=str)
    parser.add_argument('--device', help='device')
    parser.add_argument('--backend', help='backend', choices={'tf', 'pytorch'}, default='tf')
    parser.add_argument('--prefer_eager', help="If running in TensorFlow, should we prefer eager model", type=str2bool)


    args = parser.parse_known_args()[0]

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


if __name__ == '__main__':
    main()