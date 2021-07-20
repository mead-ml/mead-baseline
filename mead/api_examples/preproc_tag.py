import argparse
from pathlib import Path
import baseline
from baseline.vectorizers import BPEVectorizer1D
from eight_mile.utils import write_yaml, write_json
import json
import logging
import numpy as np
import os
from eight_mile.utils import Offsets
from eight_mile.progress import *
logger = logging.getLogger('baseline')

try:
    import tensorflow as tf
except:
    pass

def convert_to_pairs(vec, tokens, labels, label_vocab):

    for t, l in zip(tokens, labels):
        if t in Offsets.VALUES:
            yield (t, l)
        elif t == '<unk>':
            yield (Offsets.VALUES[Offsets.UNK], l)
        elif t == '<eos>':
            yield (Offsets.VALUES[Offsets.EOS], l)
        else:
            subwords = vec.tokenizer.apply([vec.transform_fn(t)])[0].split()
            subword_labels = [Offsets.PAD] * len(subwords)
            subword_labels[0] = label_vocab[l]
            for x, y in zip(subwords, subword_labels):
                yield (vec.vocab.get(x, Offsets.UNK), y)

def read_vocab_file(vocab_file):
    i2w = {}
    with open(vocab_file) as rf:
        for i, line in enumerate(rf):
            word = line.strip().split(' ')[0]
            i2w[i] = word
    return i2w

def dict_doc(f, vocab):
    docs = {}

    doc_id = None
    buffer = []
    with open(f) as rf:
        for line in rf:
            line = line.strip()
            if line.startswith("ID"):
                if doc_id:
                    docs[doc_id] = buffer
                    buffer = []
                doc_id = line.split(" ")[-1]
            else:
                buffer += [[vocab[int(x)] for x in line.split(" ")]]
        if doc_id and buffer:
            docs[doc_id] = buffer
    return docs

def annot_doc(f):
    docs = {}

    doc_id = None
    buffer = []
    with open(f) as rf:
        for line in rf:
            line = line.strip()
            if line.strip().startswith("ID"):
                if doc_id:
                    docs[doc_id] = buffer
                    buffer = []
                doc_id = line.split(" ")[-1]
            else:
                buffer += [[int(x) for x in line.split("\t")]]
        if doc_id and buffer:
            docs[doc_id] = buffer
    return docs


def create_record(chunk, str_lookup, label_lookup, prefix, suffix):
    """Emit a record

    :param chunk: A chunk of integer inputs
    :param str_lookup: A lookup table from integers to strings
    :param prefix: A prefix integer token
    :param suffix: A suffix integer token
    :return: An object with `[xy]_str` and `[xy]` entries
    """

    text_chunk, label_chunk = zip(*chunk)
    if prefix:
        text_prefix, label_prefix = prefix
        text_chunk = (text_prefix,) + text_chunk
        label_chunk = (label_prefix,) + label_chunk
    if suffix:
        text_suffix, label_suffix = suffix
        text_chunk = text_chunk + (text_suffix,)
        label_chunk = label_chunk + (label_suffix,)

    # TODO, Add BPE dropout
    inputs = text_chunk
    labels = label_chunk
    x_str = [str_lookup[s] for s in inputs]
    y_str = [label_lookup[s] for s in labels]
    return {'x': inputs, 'y': labels, 'x_str': x_str, 'y_str': y_str}


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

def makedir_if_none(d):
    if not os.path.exists(d):
        logger.info("Creating %s", d)
        os.makedirs(d)

def create_file_writer(fmt, name, fields, max_file_size_mb):
    if fmt == 'tsv':
        return TSVWriter(name, fields, max_file_size_mb)
    if fmt == 'json':
        return JSONLWriter(name, fields, max_file_size_mb)
    return TFRecordRollingWriter(name, fields, max_file_size_mb)


def write_files(annot_files, doc_files, fw, output_dir, pg_name):
    num_samples = 0
    indices2word = baseline.revlut(VECTORIZER.vocab)
    indices2labels = baseline.revlut(LABELS)
    lookup_indices = []
    pg = create_progress_bar(len(annot_files), name=pg_name)


    for annot in pg(annot_files):
        doc = os.path.join(doc_files, annot.name)
        assert (os.path.exists(doc))
        td = dict_doc(doc, DOC2WORD)
        ad = annot_doc(annot)
        # For each document
        for doc_id in ad.keys():
            yd = []
            this_doc = td[doc_id]
            for sent in this_doc:
                yd.append(['O'] * len(sent))
            this_annot = ad[doc_id]
            for annotation in this_annot:
                sid, start, end, label = annotation
                label = label2word[label]
                if (start + 1) >= end:
                    yd[sid][start] = f"S-{label}"
                else:
                    yd[sid][start] = f"B-{label}"
                    yd[sid][end - 1] = f"E-{label}"
                    for k in range(start + 1, end - 1):
                        yd[sid][k] = f"I-{label}"

            # For each document, run BPE over the whole thing
            for j, sentence in enumerate(this_doc):
                output = [pair for pair in convert_to_pairs(VECTORIZER, sentence, yd[j], LABELS)]
                available = len(output)

                while available > 0:
                    if len(lookup_indices) == NCTX:
                        record = create_record(lookup_indices, indices2word, indices2labels, PREFIX, SUFFIX)
                        fw.write(record)
                        num_samples += 1
                        lookup_indices = []
                    needed = NCTX - len(lookup_indices)
                    if available >= needed:
                        lookup_indices += output[:needed]
                        output = output[needed:]
                        available -= needed
                        record = create_record(lookup_indices, indices2word, indices2labels, PREFIX, SUFFIX)
                        fw.write(record)
                        num_samples += 1
                        lookup_indices = []
                    # The amount available is less than what we need, so read the whole thing
                    else:
                        lookup_indices += output[:available]
                        available = 0
    fw.close()
    write_yaml({'num_samples': num_samples}, os.path.join(output_dir, 'md.yml'))

def main():
    parser = argparse.ArgumentParser(description='Convert text into MLM fixed width contexts')

    parser.add_argument('--input_files',
                        help='The text to classify as a string, or a path to a file with each line as an example', type=str)
    parser.add_argument('--annot_files',
                        help='The text to classify as a string, or a path to a file with each line as an example', type=str)
    parser.add_argument('--codes', help='BPE codes')
    parser.add_argument('--vocab', help='BPE vocab')
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--fmt", type=str, default='json', choices=['json', 'tsv', 'tfrecord'])
    parser.add_argument("--fields", type=str, nargs="+", default=["x_str", "y_str"])
    parser.add_argument("--output_dir", type=str, help="Output base name, e.g. /path/to/output/record")
    parser.add_argument("--max_file_size", type=int, default=100, help="Shard size, defaults to 100MB")
    parser.add_argument("--stride", type=int, help="Tokens to stride before next read, defaults to `nctx`")
    parser.add_argument("--tok_on_eol", type=str, default="<EOS>")
    parser.add_argument("--cased", type=baseline.str2bool, default=True)
    parser.add_argument("--document_vocab", type=str, default="document.vocab")
    parser.add_argument("--label_vocab", type=str, default="label.vocab")
    parser.add_argument("--valid_split", type=float, default=0.05)
    parser.add_argument("--prefix", default="<GO>")
    parser.add_argument("--suffix", default="<EOS>")
    parser.add_argument("--pg_name", choices=["tqdm", "default"], default="default")


    args = parser.parse_args()
    annot_files = list(Path(args.annot_files).iterdir())
    valid_split = int(len(annot_files) * args.valid_split)
    VALID_FILES = annot_files[:valid_split]
    TRAIN_FILES = annot_files[valid_split:]

    VECTORIZER = BPEVectorizer1D(transform_fn=baseline.lowercase if not args.cased else lambda x: x,
                                 model_file=args.codes, vocab_file=args.vocab, mxlen=1024)
    NCTX = args.nctx - 2
    PREFIX = (VECTORIZER.vocab[args.prefix], Offsets.GO,)
    SUFFIX = (VECTORIZER.vocab[args.suffix], Offsets.EOS,)

    DOC2WORD = read_vocab_file(args.document_vocab)
    label2word = read_vocab_file(args.label_vocab)
    LABELS = {Offsets.VALUES[k]:k for k in range(Offsets.OFFSET)}
    for label in label2word.values():
        for prefix in ["B", "I", "E", "S"]:
            LABELS[f"{prefix}-{label}"] = len(LABELS)

    LABELS["O"] = len(LABELS)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    write_json(LABELS, os.path.join(args.output_dir, 'labels.json'))
    valid_dir = os.path.join(args.output_dir, 'valid')
    train_dir = os.path.join(args.output_dir, 'train')
    makedir_if_none(args.output_dir)
    makedir_if_none(train_dir)
    makedir_if_none(valid_dir)

    logger.info("Converting validation files")
    fw_valid = create_file_writer(args.fmt, os.path.join(valid_dir, 'valid'), args.fields, args.max_file_size)
    write_files(VALID_FILES, args.input_files, fw_valid, valid_dir, args.pg_name)

    logger.info("Converting training files")
    fw_train = create_file_writer(args.fmt, os.path.join(train_dir, 'train'), args.fields, args.max_file_size)
    write_files(TRAIN_FILES, args.input_files, fw_train, train_dir, args.pg_name)


if __name__ == '__main__':
    main()
