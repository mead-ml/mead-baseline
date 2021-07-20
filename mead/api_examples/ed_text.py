from __future__ import print_function
import baseline as bl
import argparse
import os
from baseline.utils import str2bool

def main():
    parser = argparse.ArgumentParser(description='Encoder-Decoder execution')
    parser.add_argument('--model', help='An encoder-decoder model', required=True, type=str)
    parser.add_argument('--text', help='raw value or a file', type=str)
    parser.add_argument('--backend', help='backend', default='tf')
    parser.add_argument('--remote', help='(optional) remote endpoint', type=str) # localhost:8500
    parser.add_argument('--name', help='(optional) signature name', type=str)
    parser.add_argument('--target', help='A file to write decoded output (or print to screen)')
    parser.add_argument('--tsv', help='print tab separated', type=bl.str2bool, default=False)
    parser.add_argument('--batchsz', help='Size of a batch to pass at once', default=32, type=int)
    parser.add_argument('--device', help='device')
    parser.add_argument('--alpha', type=float, help='If set use in the gnmt length penalty.')
    parser.add_argument('--beam', type=int, default=30, help='The size of beam to use.')
    parser.add_argument('--prefer_eager', help="If running in TensorFlow, should we prefer eager model", type=str2bool)

    args = parser.parse_known_args()[0]

    if args.backend == 'tf':
        from eight_mile.tf.layers import set_tf_eager_mode
        set_tf_eager_mode(args.prefer_eager)

    batches = []
    if os.path.exists(args.text) and os.path.isfile(args.text):
        with open(args.text, 'r') as f:
            batch = []
            for line in f:
                text = line.strip().split()
                if len(batch) == args.batchsz:
                    batches.append(batch)
                    batch = []
                batch.append(text)

            if len(batch) > 0:
                batches.append(batch)

    else:
        batch = [args.text.split()]
        batches.append(batch)

    m = bl.EncoderDecoderService.load(args.model, backend=args.backend, beam=args.beam,
                                      remote=args.remote, name=args.name, device=args.device)

    f = open(args.target, 'w') if args.target is not None else None

    for texts in batches:
        decoded = m.predict(texts, alpha=args.alpha, beam=args.beam)
        for src, dst in zip(texts, decoded):
            src_str = ' '.join(src)
            dst_str = ' '.join(dst)
            if args.tsv:
                line = src_str + '\t' + dst_str
            else:
                line = dst_str

            print(line, file=f, flush=True)

    if f is not None:
        f.close()


if __name__ == '__main__':
    main()
