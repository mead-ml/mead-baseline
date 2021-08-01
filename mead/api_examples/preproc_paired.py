import sys
import argparse
import baseline
from baseline.vectorizers import BPEVectorizer1D
from mead.api_examples.preproc_utils import *
from eight_mile.utils import (
    write_yaml, Timer
)
from typing import Optional
import numpy as np
import os


def create_record(chunk: list, str_lookup: dict, prefix: Optional[str], suffix: Optional[str], masking: Optional[Masking]=None):
    """Emit a record

    :param chunk: A chunk of integer inputs
    :param str_lookup: A lookup table from integers to strings
    :param prefix: A prefix integer token
    :param suffix: A suffix integer token
    :param mask_value: An integer value representing a [MASK]
    :param vocab_size: The total size of the vocab
    :param pad_y: Should we replace non-[MASK] X values with <PAD> in Y?
    :return: An object with `[xy]_str` and `[xy]` entries
    """
    ignore_prefix = False
    ignore_suffix = False
    if prefix:
        chunk = [prefix] + chunk
        ignore_prefix = True
    if suffix:
        chunk = chunk + [suffix]
        ignore_suffix = True

    if not masking:
        inputs = np.array(chunk)
        return {'x': inputs, 'x_str': [str_lookup[s] for s in inputs]}
    inputs, labels = masking(np.array(chunk), ignore_prefix, ignore_suffix)
    return {'x': inputs, 'y': labels, 'x_str': [str_lookup[s] for s in inputs], 'y_str': [str_lookup[s] for s in labels]}


def run(input_files=[], input_pattern='*.txt', codes=None, vocab=None, nctx=256, fmt='json', fields=['x_str', 'y_str'],
        output=None, x_prefix=None, x_suffix=None, y_prefix=None, y_suffix=None, max_file_size=100, cased=True,
        mask_type="mlm", module=None, pad_y=True, extra_tokens=['[CLS]', '[MASK]'],
        tgt_nctx=None, world_size=1, world_offset=0, **kwargs):
    timer = Timer()

    if module:
        logger.warning("Loading custom user module %s for masking rules", module)
        baseline.import_user_module(module)

    if os.path.isdir(input_files):
        import glob
        input_files = list(glob.glob(os.path.join(input_files, input_pattern)))
        if not output:
            output = os.path.join(input_files, 'records')
    else:
        input_files = [input_files]
        if not output:
            output = f'{input_files}.records'

    logger.info('Output [%s]', output)
    if not tgt_nctx:
        tgt_nctx = 64
    transform = baseline.lowercase if not cased else lambda x: x
    vectorizer = BPEVectorizer1D(transform_fn=transform, model_file=codes, vocab_file=vocab, mxlen=1024, extra_tokens=extra_tokens)

    if x_prefix:
        x_prefix = vectorizer.vocab[x_prefix]
    if x_suffix:
        x_suffix = vectorizer.vocab[x_suffix]
    if y_prefix:
        y_prefix = vectorizer.vocab[y_prefix]
    if y_suffix:
        y_suffix = vectorizer.vocab[y_suffix]

    indices2word = baseline.revlut(vectorizer.vocab)
    root_dir = os.path.dirname(output)
    masking = create_masking(mask_type, vectorizer.vocab, pad_y)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Create a file writer for this shard
    fw = create_file_writer(fmt, output, fields, max_file_size, 1000 * world_offset)
    num_read = -1
    num_samples_this_worker = 0

    for text in input_files:
        with open(text, encoding='utf-8') as rf:
            print(f"Reading from {text}...")
            for line in rf:
                num_read += 1
                if num_read % world_size != world_offset:
                    continue

                to_bpe = line.strip().split()
                if not to_bpe:
                    continue

                output, available = vectorizer.run(to_bpe, vectorizer.vocab)
                x, y = masking(output[:available], False, False)
                if x_prefix:
                    x = [x_prefix] + x
                if y_prefix:
                    y = [y_prefix] + y
                if x_suffix:
                    x += [x_suffix]
                if y_suffix:
                    y += [y_suffix]

                x = x[:nctx]
                y = y[:tgt_nctx]
                x_t = np.zeros(nctx, dtype=output.dtype)
                y_t = np.zeros(tgt_nctx, dtype=output.dtype)
                x_t[:len(x)] = x
                y_t[:len(y)] = y
                record = {'x': x_t, 'y': y_t, 'x_str': [indices2word[s] for s in x_t], 'y_str': [indices2word[s] for s in y_t]}
                if masking.is_valid(record):
                    fw.write(record)
                    num_samples_this_worker += 1

    fw.close()
    duration = timer.elapsed()
    print("Processed {:,} samples in {:.2f}s".format(num_samples_this_worker, duration))
    f_name = f'md-{world_offset}.yml' if world_size > 1 else 'md.yml'
    write_yaml({'num_samples': num_samples_this_worker}, os.path.join(root_dir, f_name))


def main():
    argv = sys.argv[1:]
    args = parse_args(argv)
    run(**vars(args))


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Convert paired text into fixed width contexts')
    parser.add_argument('--input_files',
                        help='The text to convert to LM or a path to a file with each line as an example', type=str)
    parser.add_argument('--input_pattern', type=str, default='*.txt')
    parser.add_argument('--world_size', type=int, default=1, help="Can be used as decimation factor, or to support multiproc")
    parser.add_argument('--world_offset', type=int, default=0, help="Offset for decimation or processor")
    parser.add_argument('--codes', help='BPE codes')
    parser.add_argument('--vocab', help='BPE vocab')
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--fmt", type=str, default='json', choices=['json', 'tsv', 'tfrecord'])
    parser.add_argument("--fields", type=str, nargs="+", default=["x_str", "y_str"])
    parser.add_argument("--output", type=str, help="Output base name, e.g. /path/to/output/record")
    parser.add_argument("--x_prefix", type=str, help="Prefix every x with this token")
    parser.add_argument("--x_suffix", type=str, help="Suffix every x with this token")
    parser.add_argument("--y_prefix", type=str, help="Prefix every y with this token")
    parser.add_argument("--y_suffix", type=str, help="Suffix every y with this token")
    parser.add_argument("--max_file_size", type=int, default=100, help="Shard size, defaults to 100MB")
    parser.add_argument("--cased", type=baseline.str2bool, default=True)
    parser.add_argument("--mask_type", type=str, default="mlm", help="Masking rules, including 'mlm' and 'causal'")
    parser.add_argument("--module", default=None, help="Module containing custom masking rules")
    parser.add_argument("--pad_y", type=baseline.str2bool, default=True,
                        help="Replace all non-masked Y values with <PAD>")
    parser.add_argument("--extra_tokens", type=str, nargs="+", default=['[CLS]', '[MASK]'])
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main()
