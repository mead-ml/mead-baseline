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
    parser.add_argument('--backend', help='backend', choices={'tf', 'pytorch', 'onnx'}, default='tf')
    parser.add_argument('--remote', help='(optional) remote endpoint, normally localhost:8500', type=str) # localhost:8500
    parser.add_argument('--name', help='(optional) service name as the server may serves multiple models', type=str)
    parser.add_argument('--device', help='device')
    parser.add_argument('--preproc', help='(optional) where to perform preprocessing', choices={'client', 'server'},
                        default='client')
    parser.add_argument('--batchsz', help='batch size when --text is a file', default=100, type=int)
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--modules', default=[], nargs="+")
    parser.add_argument('--prefer_eager', help="If running in TensorFlow, should we prefer eager model", type=str2bool, default=False)
    parser.add_argument('--scores', '-s', action="store_true")
    args = parser.parse_args()

    if args.backend == 'tf':
        from eight_mile.tf.layers import set_tf_eager_mode
        set_tf_eager_mode(args.prefer_eager)

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
    batched = [texts[i:i + args.batchsz] for i in range(0, len(texts), args.batchsz)]

    m = bl.ClassifierService.load(args.model, backend=args.backend, remote=args.remote,
                                  name=args.name, preproc=args.preproc,
                                  device=args.device, model_type=args.model_type)
    for texts in batched:
        for text, output in zip(texts, m.predict(texts)):
            if args.scores:
                print("{}, {}".format(" ".join(text), output))
            else:
                print("{}, {}".format(" ".join(text), output[0][0]))


if __name__ == '__main__':
    main()
