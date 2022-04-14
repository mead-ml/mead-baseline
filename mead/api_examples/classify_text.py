import baseline as bl
import argparse
import os
import json


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
    parser.add_argument('--is_dual', action="store_true")
    parser.add_argument('--label_first', action="store_true", help="Use the second column")
    parser.add_argument("--output_delim", default="\t")
    parser.add_argument("--output_type", default="tsv", choices=["tsv", "json"])
    parser.add_argument("--no_text_output", action="store_true", help="Dont write the text")
    args = parser.parse_args()

    for mod_name in args.modules:
        bl.import_user_module(mod_name)

    labels = []
    if os.path.exists(args.text) and os.path.isfile(args.text):
        texts = []
        with open(args.text, 'r') as f:
            for line in f:
                text = line.strip().split('\t')
                if args.label_first:
                    labels.append(text[0])
                    text = text[1:]
                if args.is_dual:
                    first = text[0].split()
                    second = text[1].split()
                    text = [first, second]
                else:
                    text = text[0].split()

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
            if args.scores:
                guess_output = output
            else:
                guess_output = output[0][0]

            if args.output_type == 'tsv':
                if args.no_text_output:
                    text_output = ''
                else:
                    text_output = ' '.join(text) + args.output_delim

                s = f"{text_output}{guess_output}"
                if args.label_first:
                    s = f"{next(label_iter)}{args.output_delim}{s}"
            else:
                if args.is_dual:
                    text_output = [' '.join(text[0]), ' '.join(text[1])]
                else:
                    text_output = ' '.join(text)
                if args.scores:
                    guess_output = {kv[0]: kv[1] for kv in guess_output}
                json_output = {'prediction': guess_output}
                if not args.no_text_output:
                    json_output['text'] = text_output
                if args.label_first:
                    json_output['label'] = next(label_iter)
                s = json.dumps(json_output)
            print(s)


if __name__ == '__main__':
    main()
