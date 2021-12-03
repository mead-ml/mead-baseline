import baseline as bl
import argparse
import os
from eight_mile.utils import str2bool


def main():
    parser = argparse.ArgumentParser(description='Classify text with a model')
    parser.add_argument('--model', help='The path to either the .zip file created by training or to the client bundle '
                                        'created by exporting', required=True, type=str)
    parser.add_argument('--text', help='The text to classify as a string, or a path to a file with each line as an example',
                        type=str)
    parser.add_argument('--backend', help='backend', choices={'tf', 'pytorch', 'onnx'}, default='pytorch')
    parser.add_argument('--remote', help='(optional) remote endpoint, normally localhost:8500', type=str) # localhost:8500
    parser.add_argument('--name', help='(optional) service name as the server may serves multiple models', type=str)
    parser.add_argument('--device', help='device')
    parser.add_argument('--preproc', help='(optional) where to perform preprocessing', choices={'client', 'server'},
                        default='client')
    parser.add_argument('--batchsz', help='batch size when --text is a file', default=100, type=int)
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--modules', default=[], nargs="+")
    parser.add_argument('--scores', '-s', action="store_true")
    parser.add_argument('--label_first', action="store_true", help="Use the second column")
    parser.add_argument("--output_delim", default="\t")
    parser.add_argument("--no_text_output", action="store_true", help="Dont write the text")
    args = parser.parse_args()

    for mod_name in args.modules:
        bl.import_user_module(mod_name)

    labels = []
    if os.path.exists(args.text) and os.path.isfile(args.text):
        texts = []
        with open(args.text, 'r') as f:
            for line in f:
                text = line.strip().split()
                if args.label_first:
                    labels.append(text[0])
                    text = text[1:]
                texts += [text]

    else:
        texts = [args.text.split()]
    batched = [texts[i:i + args.batchsz] for i in range(0, len(texts), args.batchsz)]

    m = bl.ClassifierService.load(args.model, backend=args.backend, remote=args.remote,
                                  name=args.name, preproc=args.preproc,
                                  device=args.device, model_type=args.model_type)

    if args.label_first:
        label_iter = iter(labels)
    for texts in batched:
        for text, output in zip(texts, m.predict(texts)):

            if args.no_text_output:
                text_output = ''
            else:
                text_output = ' '.join(text) + {args.output_delim}
            if args.scores:
                guess_output = output
            else:
                guess_output = output[0][0]

            s = f"{text_output}{guess_output}"
            if args.label_first:
                s = f"{next(label_iter)}{args.output_delim}{s}"
            print(s)

if __name__ == '__main__':
    main()
