import os
import argparse
import baseline as bl

parser = argparse.ArgumentParser(description='Score text with a language model.')
parser.add_argument("--model", help='The path to either the .zip file (or dir) created by training', required=True)
parser.add_argument("--text", help="The text to score as a string, or a path to a file", required=True)
parser.add_argument("--backend", default="tf", help="the dl backend framework the model was trained with", choices=("tensorflow", "tf", "pytorch", "pyt"))
parser.add_argument("--device", help="the device to run the model on")
parser.add_argument("--prob", default="joint", choices=("joint", "conditional"), help="Should you score the whole string (joint) or the score of the last token given the previous (conditional)")
args = parser.parse_args()

if os.path.exists(args.text) and os.path.isfile(args.text):
    with open(args.text, 'r') as f:
        text = [f.read.strip().split()]
else:
    text = [args.text.split()]

m = bl.LanguageModelService.load(args.model, backend=args.backend, device=args.device)
scores = m.score(text, prob=args.prob)

if args.prob == 'joint':
    print(f"P({' '.join(text[0])}) = {scores[0]}")
else:
    print(f"P({text[0][-1]} | {' '.join(text[0][:-1])}) = {scores[0]}")
