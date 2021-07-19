import argparse
import baseline
from baseline.vectorizers import BPEVectorizer1D
from eight_mile.utils import (
    mlm_masking,
    write_yaml,
    register,
    optional_params,
)
import json
from typing import Optional
import logging
import numpy as np
import os
logger = logging.getLogger('baseline')
try:
    import tensorflow as tf
except:
    pass


MASKING_RULE_DEFS = {}


@optional_params
def register_masking(cls, name=None):
    """Register a class as a handler for masking rules by key name"""
    return register(cls, MASKING_RULE_DEFS, name, "masking rule defs")

def create_masking(mask_type, vocab, pad_y):
    Constructor = MASKING_RULE_DEFS.get(mask_type)
    if not Constructor:
        logger.warning("No masking algorithm for %s, treating as causal", mask_type)
        return None
    logger.info("Creating constructor %s for mask_type %s", str(Constructor), mask_type)
    masking = Constructor(vocab, pad_y=pad_y)
    return masking


class Masking:
    def __init__(self):
        pass

    def __call__(self, chunk: np.ndarray, ignore_prefix: bool, ignore_suffix: bool):
        pass


@register_masking("mlm")
class MaskMLM(Masking):
    def __init__(self, vocab, pad_y):
        vocab_size = max(vocab.values()) + 1
        mask_value = vocab['[MASK]']
        self.mask_value = mask_value
        self.vocab_size = vocab_size

        self.pad_y = pad_y

    def __call__(self, chunk: np.ndarray, ignore_prefix: bool, ignore_suffix: bool):

        return mlm_masking(chunk, self.mask_value, self.vocab_size, ignore_prefix, ignore_suffix, pad_y=self.pad_y)


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


def in_bytes(mb):
    return mb * 1024 * 1024


class RollingWriter:
    def __init__(self, name, fields, max_file_size_mb):
        self.name = name
        self.counter = 1
        self.fields = fields
        self.current_file_size = 0
        self.writer = None
        self.max_file_size = in_bytes(max_file_size_mb)
        self._rollover_file()

    def _open_file(self, filename):
        return open(filename, 'w')

    def _rollover_file(self):
        if self.writer:
            self.writer.close()
        filename = f'{self.name}-{self.counter}.{self.suffix}'
        self.counter += 1
        self.current_file_size = 0
        logger.info("Rolling over.  New file [%s]", filename)
        self.writer = self._open_file(filename)

    @property
    def suffix(self):
        raise Exception("Dont know suffix in ABC")

    def _write_line(self, str_val):
        self.writer.write(str_val)
        return len(str_val.encode("utf8"))

    def _write_line_rollover(self, l):
        sz = self._write_line(l)
        self.current_file_size += sz
        if self.current_file_size > self.max_file_size:
            self._rollover_file()

    def close(self):
        self.writer.close()


class TSVWriter(RollingWriter):
    def __init__(self, name, fields, max_file_size_mb):
        super().__init__(name, fields, max_file_size_mb)

    def _to_str(self, value):
        if isinstance(value, np.ndarray):
            value = [str(v) for v in value]
        return value

    def write(self, record):
        l = [' '.join(self._to_str(record[f])) for f in self.fields]
        str_val = '\t'.join(l) + '\n'
        self._write_line_rollover(str_val)

    @property
    def suffix(self):
        return 'tsv'


class JSONLWriter(RollingWriter):

    def __init__(self, name, fields, max_file_size_mb):
        super().__init__(name, fields, max_file_size_mb)

    def write(self, record):
        r = {}
        for f in self.fields:
            if isinstance(record[f], np.ndarray):
                value = record[f].tolist()
            else:
                value = record[f]
            r[f] = value
        output = json.dumps(r) + '\n'
        self._write_line_rollover(output)

    @property
    def suffix(self):
        return 'json'


class TFRecordRollingWriter(RollingWriter):
    def __init__(self, name, fields, max_file_size_mb):
        try:
            self.RecordWriterClass = tf.io.TFRecordWriter
        except Exception as e:
            raise Exception("tfrecord package could not be loaded, pip install that first, along with crc32c")
        super().__init__(name, fields, max_file_size_mb)

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _open_file(self, filename):
        return self.RecordWriterClass(filename)

    def _write_line(self, str_val):
        self.writer.write(str_val)
        return len(str_val)

    def serialize_tf_example(self, record):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature= {}
        for f in self.fields:
            if f.endswith('_str'):
                value = ' '.join(record[f])
                value = TFRecordRollingWriter._bytes_feature(value.encode('utf-8'))
            else:
                value = TFRecordRollingWriter._int64_feature(record[f])
            feature[f] = value

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def write(self, record):
        example_str = self.serialize_tf_example(record)
        self._write_line_rollover(example_str)

    @property
    def suffix(self):
        return 'tfrecord'


def create_file_writer(fmt, name, fields, max_file_size_mb):
    if fmt == 'tsv':
        return TSVWriter(name, fields, max_file_size_mb)
    if fmt == 'json':
        return JSONLWriter(name, fields, max_file_size_mb)
    return TFRecordRollingWriter(name, fields, max_file_size_mb)


def main():
    parser = argparse.ArgumentParser(description='Convert text into LM fixed width contexts')

    parser.add_argument('--input_files',
                        help='The text to convert to LM or a path to a file with each line as an example', type=str)
    parser.add_argument('--input_pattern', type=str, default='*.txt')
    parser.add_argument('--codes', help='BPE codes')
    parser.add_argument('--vocab', help='BPE vocab')
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--fmt", type=str, default='json', choices=['json', 'tsv', 'tfrecord'])
    parser.add_argument("--fields", type=str, nargs="+", default=["x_str", "y_str"])
    parser.add_argument("--output", type=str, help="Output base name, e.g. /path/to/output/record")
    parser.add_argument("--prefix", type=str, help="Prefix every line with this token")
    parser.add_argument("--suffix", type=str, help="Suffix every line with this token")
    parser.add_argument("--max_file_size", type=int, default=100, help="Shard size, defaults to 100MB")
    parser.add_argument("--stride", type=int, help="Tokens to stride before next read, defaults to `nctx`")
    parser.add_argument("--tok_on_eol", type=str, default="<EOS>")
    parser.add_argument("--cased", type=baseline.str2bool, default=True)
    parser.add_argument("--mask_type", type=str, default="mlm", help="Masking rules, including 'mlm' and 'causal'")
    parser.add_argument("--pad_y", type=baseline.str2bool, default=True, help="Replace all non-masked Y values with <PAD>")
    parser.add_argument("--extra_tokens", type=str, nargs="+", default=['[CLS]', '[MASK]'])
    args = parser.parse_args()

    if os.path.isdir(args.input_files):
        import glob
        input_files = list(glob.glob(os.path.join(args.input_files, args.input_pattern)))
        if not args.output:
            args.output = os.path.join(args.input_files, 'records')
    else:
        input_files = [args.input_files]
        if not args.output:
            args.output = f'{args.input_files}.records'

    print(args.output)
    transform = baseline.lowercase if not args.cased else lambda x: x
    vectorizer = BPEVectorizer1D(transform_fn=transform, model_file=args.codes, vocab_file=args.vocab, mxlen=1024, extra_tokens=args.extra_tokens)

    lookup_indices = []
    words = []
    indices2word = baseline.revlut(vectorizer.vocab)
    nctx = args.nctx
    prefix = suffix = None
    root_dir = os.path.dirname(args.output)
    masking = create_masking(args.mask_type, vectorizer.vocab, args.pad_y)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if args.prefix:
        nctx -= 1
        prefix = vectorizer.vocab[args.prefix]

    if args.suffix:
        nctx -= 1
        suffix = vectorizer.vocab[args.suffix]

    fw = create_file_writer(args.fmt, args.output, args.fields, args.max_file_size)
    num_samples = 0
    for text in input_files:
        with open(text, encoding='utf-8') as rf:
            print(f"Reading from {text}...")
            for line in rf:
                to_bpe = line.strip().split()
                if not to_bpe:
                    continue
                to_bpe += [args.tok_on_eol]

                output, available = vectorizer.run(to_bpe, vectorizer.vocab)
                while available > 0:
                    if len(lookup_indices) == nctx:
                        record = create_record(lookup_indices, indices2word, prefix, suffix, masking=masking)
                        fw.write(record)
                        num_samples += 1
                        lookup_indices = []
                    needed = nctx - len(lookup_indices)
                    if available >= needed:
                        lookup_indices += output[:needed].tolist()
                        output = output[needed:]
                        available -= needed
                        record = create_record(lookup_indices, indices2word, prefix, suffix, masking=masking)
                        fw.write(record)
                        num_samples += 1
                        lookup_indices = []
                    # The amount available is less than what we need, so read the whole thing
                    else:
                        lookup_indices += output[:available].tolist()
                        available = 0

    fw.close()
    write_yaml({'num_samples': num_samples}, os.path.join(root_dir, 'md.yml'))


if __name__ == '__main__':
    main()
