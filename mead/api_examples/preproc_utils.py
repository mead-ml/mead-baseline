import numpy as np
from eight_mile.utils import register, mlm_masking, optional_params
import logging
import json
try:
    import tensorflow as tf
except:
    pass

logger = logging.getLogger('baseline')


MASKING_RULE_DEFS = {}


@optional_params
def register_masking(cls, name=None):
    """Register a class as a handler for masking rules by key name"""
    r = register(cls, MASKING_RULE_DEFS, name, "masking rule defs")
    return r


def create_masking(mask_type, vocab, pad_y):
    print('creating', MASKING_RULE_DEFS)
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

    def is_valid(self, record: dict) -> bool:
        return True


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


def in_bytes(mb):
    return mb * 1024 * 1024


class RollingWriter:
    def __init__(self, name, fields, max_file_size_mb, counter=1):
        self.name = name
        self.counter = counter
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
    def __init__(self, name, fields, max_file_size_mb, counter=1):
        try:
            self.RecordWriterClass = tf.io.TFRecordWriter
        except Exception as e:
            raise Exception("tfrecord package could not be loaded, pip install that first, along with crc32c")
        super().__init__(name, fields, max_file_size_mb, counter)

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


def create_file_writer(fmt, name, fields, max_file_size_mb, counter=1):
    ClassDef = TFRecordRollingWriter
    if fmt == 'tsv':
        ClassDef = TSVWriter
    elif fmt.startswith('json'):
        ClassDef = JSONLWriter
    return ClassDef(name, fields, max_file_size_mb, counter)


