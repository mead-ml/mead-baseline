import os
import argparse
import baseline as bl

def main():
    parser = argparse.ArgumentParser(description='Score text with a language model.')
    parser.add_argument("--model", help='The path to either the .zip file (or dir) created by training', required=True)
    parser.add_argument("--text", help="The text to score as a string, or a path to a file", required=True)
    parser.add_argument("--backend", default="tf", help="the dl backend framework the model was trained with", choices=("tensorflow", "tf", "pytorch", "pyt"))
    parser.add_argument("--device", help="the device to run the model on")
    parser.add_argument("--prob", default="conditional", choices=("joint", "conditional"), help="Should you score the whole string (joint) or the score of the last token given the previous (conditional)")
    parser.add_argument('--prefer_eager', help="If running in TensorFlow, should we prefer eager model", type=bl.str2bool, default=False)
    args = parser.parse_args()

    if args.backend == 'tf':
        from eight_mile.tf.layers import set_tf_eager_mode
        set_tf_eager_mode(args.prefer_eager)

    if os.path.exists(args.text) and os.path.isfile(args.text):
        with open(args.text, 'r') as f:
            text = [f.read.strip().split()]
    else:
        text = [args.text.split()]

    if len(text) > 1:
        raise ValueError(f"Currently only batch size 1 supported for LanguageModelServices, got {len(text)}")

    m = bl.LanguageModelService.load(args.model, backend=args.backend, device=args.device)

    if args.prob == 'joint':
        scores = m.joint(text)
        print(f"P({' '.join(text[0])}) = {scores[0]}")
    else:
        *context, targets = list(zip(*text))
        context = list(zip(*context))
        scores = m.conditional(context, target=targets)
        print(f"P({text[0][-1]} | {' '.join(text[0][:-1])}) = {scores[0]}")


if __name__ == '__main__':
    main()
