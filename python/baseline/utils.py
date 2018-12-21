import six

import os
import sys
import json
import hashlib
import logging
import zipfile
import platform
import importlib
from operator import lt, le, gt, ge
from contextlib import contextmanager
from functools import partial, update_wrapper, wraps
import numpy as np
import addons
import collections

__all__ = []


def optional_params(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        return lambda x: func(x, *args, **kwargs)
    return wrapped


def parameterize(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        return lambda x: func(x, *args, **kwargs)
    return decorator


@parameterize
def export(obj, all_list=None):
    """Add a function or class to the __all__.

    When exporting something with out using as a decorator do it like so:
        `func = exporter(func)`
    """
    all_list.append(obj.__name__)
    return obj


exporter = export(__all__)


@exporter
class Offsets:
    """Support pre 3.4"""
    PAD, GO, EOS, UNK, OFFSET = range(0, 5)
    VALUES = ["<PAD>", "<GO>", "<EOS>", "<UNK>"]


@exporter
def register(cls, registry, name=None, error=''):
    if name is None:
        name = cls.__name__
    if name in registry:
        raise Exception('Error: attempt to re-define previously registered {} {} (old: {}, new: {})'.format(error, name, registry[name], cls))
    if hasattr(cls, 'create'):
        registry[name] = cls.create
    else:
        registry[name] = cls
    return cls


@contextmanager
def redirect(from_stream, to_stream):
    original_from = from_stream.fileno()
    saved_from = os.dup(original_from)
    os.dup2(to_stream.fileno(), original_from)
    try:
        yield
        os.dup2(saved_from, original_from)
    except Exception as e:
        os.dup2(saved_from, original_from)
        raise(e)


@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull, redirect(sys.stdout, devnull), redirect(sys.stderr, devnull):
        yield


class JSONFormatter(logging.Formatter):
    def format(self, record):
        try:
            if isinstance(record.msg, (list, dict)):
                return json.dumps(record.msg)
        except TypeError:
            pass
        return super(JSONFormatter, self).format(record)


@exporter
class Colors(object):
    GREEN = '\033[32;1m'
    RED = '\033[31;1m'
    YELLOW = '\033[33;1m'
    BLACK = '\033[30;1m'
    CYAN = '\033[36;1m'
    RESTORE = '\033[0m'


@exporter
def color(msg, color):
    if platform.system() == 'Windows':
        return msg
    return u"{}{}{}".format(color, msg, Colors.RESTORE)


@exporter
def sequence_mask(lengths, max_len=-1):
    if max_len < 0:
        max_len = np.max(lengths)
    row = np.arange(0, max_len).reshape(1, -1)
    col = np.reshape(lengths, (-1, 1))
    return (row < col).astype(np.uint8)


@exporter
def transition_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a CRF mask.

    Returns a mask with invalid moves as 0 and valid as 1.

    :param vocab: dict, Label vocabulary mapping name to index.
    :param span_type: str, The sequence labeling formalism {IOB, IOB2, BIO, or IOBES}
    :param s_idx: int, What is the index of the GO symbol?
    :param e_idx: int, What is the index of the EOS symbol?
    :param pad_idx: int, What is the index of the PAD symbol?

    Note:
        In this mask the PAD symbol is between the last symbol and EOS, PADS can
        only move to pad and the EOS. Any symbol that can move to an EOS can also
        move to a pad.
    """
    rev_lut = {v: k for k, v in vocab.items()}
    start = rev_lut[s_idx]
    end = rev_lut[e_idx]
    pad = None if pad_idx is None else rev_lut[pad_idx]
    if span_type.upper() == "IOB":
        mask = iob_mask(vocab, start, end, pad)
    if span_type.upper() == "IOB2" or span_type.upper() == "BIO":
        mask = iob2_mask(vocab, start, end, pad)
    if span_type.upper() == "IOBES":
        mask = iobes_mask(vocab, start, end, pad)
    return mask


def iob_mask(vocab, start, end, pad=None):
    small = 0
    mask = np.ones((len(vocab), len(vocab)), dtype=np.float32)
    for from_ in vocab:
        for to in vocab:
            # Can't move to start
            if to is start:
                mask[vocab[to], vocab[from_]] = small
            # Can't move from end
            if from_ is end:
                mask[vocab[to], vocab[from_]] = small
            # Can only move from pad to pad or end
            if from_ is pad:
                if not(to is pad or to is end):
                    mask[vocab[to], vocab[from_]] = small
            elif from_ is start:
                # Can't move from start to a B
                if to.startswith("B-"):
                    mask[vocab[to], vocab[from_]] = small
            else:
                if from_.startswith("B-"):
                    # Can't move from a B to a B of another type
                    if to.startswith("B-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("I-"):
                    # Can't move from an I to a B of another type
                    if to.startswith("B-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("O"):
                    # Can't move from an O to a B
                    if to.startswith("B-"):
                        mask[vocab[to], vocab[from_]] = small
    return mask


def iob2_mask(vocab, start, end, pad=None):
    small = 0
    mask = np.ones((len(vocab), len(vocab)), dtype=np.float32)
    for from_ in vocab:
        for to in vocab:
            # Can't move to start
            if to is start:
                mask[vocab[to], vocab[from_]] = small
            # Can't move from end
            if from_ is end:
                mask[vocab[to], vocab[from_]] = small
            # Can only move from pad to pad or end
            if from_ is pad:
                if not(to is pad or to is end):
                    mask[vocab[to], vocab[from_]] = small
            elif from_ is start:
                # Can't move from start to a I
                if to.startswith("I-"):
                    mask[vocab[to], vocab[from_]] = small
            else:
                if from_.startswith("B-"):
                    # Can't move from a B to an I of a different type
                    if to.startswith("I-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("I-"):
                    # Can't move from an I to an I of another type
                    if to.startswith("I-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("O"):
                    # Can't move from an O to an I
                    if to.startswith("I-"):
                        mask[vocab[to], vocab[from_]] = small
    return(mask)


def iobes_mask(vocab, start, end, pad=None):
    small = 0
    mask = np.ones((len(vocab), len(vocab)), dtype=np.float32)
    for from_ in vocab:
        for to in vocab:
            # Can't move to start
            if to is start:
                mask[vocab[to], vocab[from_]] = small
            # Can't move from end
            if from_ is end:
                mask[vocab[to], vocab[from_]] = small
            # Can only move from pad to pad or to end
            if from_ is pad:
                if not(to is pad or to is end):
                    mask[vocab[to], vocab[from_]] = small
            elif from_ is start:
                # Can't move from start to I or E
                if to.startswith("I-") or to.startswith("E-"):
                    mask[vocab[to], vocab[from_]] = small
            else:
                if from_.startswith("B-"):
                    # Can't move from B to B, S, O, End, or Pad
                    if (
                        to.startswith("B-") or
                        to.startswith("S-") or
                        to.startswith("O") or
                        to is end or
                        to is pad
                    ):
                        mask[vocab[to], vocab[from_]] = small
                    # Can only move to matching I or E
                    elif to.startswith("I-") or to.startswith("E-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif from_.startswith("I-"):
                    # Can't move from I to B, S, O, End or Pad
                    if (
                        to.startswith("B-") or
                        to.startswith("S-") or
                        to.startswith("O") or
                        to is end or
                        to is pad
                    ):
                        mask[vocab[to], vocab[from_]] = small
                    # Can only move to matching I or E
                    elif to.startswith("I-") or to.startswith("E-"):
                        from_type = from_.split("-")[1]
                        to_type = to.split("-")[1]
                        if from_type != to_type:
                            mask[vocab[to], vocab[from_]] = small
                elif (
                    from_.startswith("E-") or
                    from_.startswith("I-") or
                    from_.startswith("S-") or
                    from_.startswith("O")
                ):
                    # Can't move from E to I or E
                    # Can't move from I to I or E
                    # Can't move from S to I or E
                    # Can't move from O to I or E
                    if to.startswith("I-") or to.startswith("E-"):
                        mask[vocab[to], vocab[from_]] = small
    return mask


@exporter
def get_version(pkg):
    s = '.'.join(pkg.__version__.split('.')[:2])
    return float(s)

@exporter
def revlut(lut):
    return {v: k for k, v in lut.items()}


@exporter
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


@exporter
def lowercase(x):
    return x.lower()


UNREP_EMOTICONS = (
    ':)',
    ':(((',
    ':D',
    '=)',
    ':-)',
    '=(',
    '(=',
    '=[[',
)

@exporter
def web_cleanup(word):
    if word.startswith('http'): return 'URL'
    if word.startswith('@'): return '@@@@'
    if word.startswith('#'): return '####'
    if word == '"': return ','
    if word in UNREP_EMOTICONS: return ';)'
    if word == '<3': return '&lt;3'
    return word


@exporter
def is_sequence(x):
    import collections
    if isinstance(x, six.string_types):
        return False
    return isinstance(x, collections.Sequence)


@exporter
def listify(x):
    """Take a scalar or list and make it a list iff not already a sequence or numpy array

    :param x: The input to convert
    :return: A list
    """
    if is_sequence(x) or isinstance(x, np.ndarray):
        return x
    return [x] if x is not None else []


@exporter
def read_json(filepath, default_value=None, strict=False):
    """Read a JSON file in.  If no file is found and default value is set, return that instead.  Otherwise error

    :param filepath: str, A file to load
    :param default_value: If the file doesn't exist, return return this. Defaults to an empty dict.
    :param strict: bool, If true raise an error on file not found.

    :return: dict, The read JSON object
    """
    if not os.path.exists(filepath):
        if strict:
            raise IOError('No file {} found'.format(filepath))
        return default_value if default_value is not None else {}
    with open(filepath) as f:
        return json.load(f)


@exporter
def read_yaml(filepath, default_value=None, strict=False):
    """Read a JSON file in.  If no file is found and default value is set, return that instead.  Otherwise error

    :param filepath: str, A file to load
    :param default_value: If the file doesn't exist, return return this. Defaults to an empty dict.
    :param strict: bool, If true raise an error on file not found.

    :return: dict, The read yaml object
    """
    if not os.path.exists(filepath):
        if strict:
            raise IOError('No file {} found'.format(filepath))
        return default_value if default_value is not None else {}
    with open(filepath) as f:
        import yaml
        return yaml.load(f)


@exporter
def read_config_file(config_file):
    """Read a config file. This method optionally supports YAML, if the dependency was already installed.  O.W. JSON plz

    :param config_file: (``str``) A path to a config file which should be a JSON file, or YAML if pyyaml is installed
    :return: (``dict``) An object
    """
    if config_file.endswith('.yml'):
        return read_yaml(config_file, strict=True)
    return read_json(config_file, strict=True)


@exporter
def read_config_stream(config_stream):
    """Read a config stream.  This may be a path to a YAML or JSON file, or it may be a str containing JSON or the name
    of an env variable, or even a JSON object directly

    :param config_stream:
    :return:
    """
    if isinstance(config_stream, (dict, list)) or config_stream is None:
        return config_stream
    if os.path.exists(config_stream) and os.path.isfile(config_stream):
        return read_config_file(config_stream)
    config = config_stream
    if config_stream.startswith("$"):
        print('Reading config from {}'.format(config_stream))
        config = os.getenv(config_stream[1:])
    return json.loads(config)


def write_sentence_conll(handle, sentence, gold, txt, idx2label):

    if len(txt) != len(sentence):
        txt = txt[:len(sentence)]

    try:
        for word, truth, guess in zip(txt, gold, sentence):
            handle.write('%s %s %s\n' % (word['text'], idx2label[truth], idx2label[guess]))
        handle.write('\n')
    except:
        print('ERROR: Failed to write lines... closing file')
        handle.close()


@exporter
def write_json(content, filepath):
    with open(filepath, "w") as f:
        json.dump(content, f, indent=True)


@exporter
def ls_props(thing):
    """List all of the properties on some object

    :param thing: Some object
    :return: The list of properties
    """
    return [x for x in dir(thing) if isinstance(getattr(type(thing), x, None), property)]


@exporter
def import_user_module(module_name):
    """Load a module that is in the python path

    :param model_name: (``str``) - the name of the module
    :return:
    """
    sys.path.append(os.path.dirname(os.path.realpath(addons.__file__)))
    mod = importlib.import_module(module_name)
    return mod


@exporter
def get_model_file(task, platform, basedir=None):
    """Model name file helper to abstract different DL platforms (FWs)

    :param dictionary:
    :param task:
    :param platform:
    :return:
    """
    basedir = './' if basedir is None else basedir
    base = '{}/{}-model'.format(basedir, task)
    rid = os.getpid()
    if platform.startswith('pyt'):
        name = '%s-%d.pyt' % (base, rid)
    else:
        name = '%s-%s-%d' % (base, platform, rid)
    print('model file [%s]' % name)
    return name


@exporter
def lookup_sentence(rlut, seq, reverse=False, padchar=''):
    """Lookup a sentence by id and return words

    :param rlut: an index -> word lookup table
    :param seq: A temporal sequence
    :param reverse: (``bool``) Should reverse?
    :param padchar: What padding character to use when replacing with words
    :return:
    """
    s = seq[::-1] if reverse else seq
    res = []
    for idx in s:
        idx = int(idx)
        char = padchar
        if idx == Offsets.EOS: break
        if idx != Offsets.PAD and idx != Offsets.GO:
            char = rlut[idx]
        res.append(char)
    return (' '.join(res)).strip()


@exporter
def topk(k, probs):
    """Get a sparse index (dictionary of top values)."""
    idx = np.argpartition(probs, probs.size-k)[-k:]
    sort = idx[np.argsort(probs[idx])][::-1]
    return dict(zip(sort, probs[sort]))



@exporter
def beam_multinomial(k, probs):
    """Prune all elements in a large probability distribution below the top K.
    
    Renormalize the distribution with only top K, and then sample n times out of that.
    """

    tops = topk(k, probs)
    i = 0
    n = len(tops.keys())
    ary = np.zeros((n))
    idx = []
    for abs_idx, v in tops.items():
        ary[i] = v
        idx.append(abs_idx)
        i += 1

    ary /= np.sum(ary)
    sample_idx = np.argmax(np.random.multinomial(1, ary))
    return idx[sample_idx]


@exporter
def fill_y(nc, yidx):
    """Convert a `B` sparse array to a dense one, to expand labels 
    
    :param nc: (``int``) The number of labels
    :param yidx: The sparse array of the labels
    :return: A dense array
    """
    xidx = np.arange(0, yidx.shape[0], 1)
    dense = np.zeros((yidx.shape[0], nc), dtype=int)
    dense[xidx, yidx] = 1
    return dense


@exporter
def convert_iob_to_bio(ifile, ofile):
    """Convert from IOB to BIO (IOB2)

    This code is copied from Xuezhe Ma (though I added comments)
    https://github.com/XuezheMax/NeuroNLP2/issues/9

    :param ifile: Original IOB format CONLL file
    :param ofile: BIO/IOB2 format
    """
    with open(ifile, 'r') as reader, open(ofile, 'w') as writer:
        prev = 'O'
        for line in reader:
            line = line.strip()
            if len(line) == 0:
                prev = 'O'
                writer.write('\n')
                continue

            tokens = line.split()
            # print tokens
            label = tokens[-1]
            # If this label is B or I and not equal to the previous
            if label != 'O' and label != prev:
                # If the last was O, it has to be a B
                if prev == 'O':
                    label = 'B-' + label[2:]
                # Otherwise if the tags are different, it also has to be a B
                elif label[2:] != prev[2:]:
                    label = 'B-' + label[2:]

            writer.write(' '.join(tokens[:-1]) + ' ' + label + '\n')
            prev = tokens[-1]


@exporter
def convert_bio_to_iobes(ifile, ofile):

    with open(ifile, 'r') as reader, open(ofile, 'w') as writer:
        lines = [line.strip() for line in reader]
        for i, line in enumerate(lines):

            tokens = line.split()
            if len(tokens) == 0:
                writer.write('\n')
                continue

            label = tokens[-1]

            if i + 1 != len(lines):
                next_tokens = lines[i+1].split()
                if len(next_tokens) > 1:
                     next_tag = next_tokens[-1]
                else:
                    next_tag = None

            # Nothing to do for label == 'O'
            if label == 'O':
                updated_label = label

            # It could be S
            elif label[0] == 'B':
                if next_tag and next_tag[0] == 'I' and next_tag[2:] == label[2:]:
                    updated_label = label
                else:
                    updated_label = label.replace('B-', 'S-')

            elif label[0] == 'I':
                if next_tag and next_tag[0] == 'I':
                    updated_label = label
                else:
                    updated_label = label.replace('I-', 'E-')
            else:
                raise Exception('Invalid IOBES format!')

            writer.write(' '.join(tokens[:-1]) + ' ' + updated_label + '\n')
            prev = tokens[-1]


@exporter
def to_spans(sequence, lut, span_type, verbose=False):
    """Turn a sequence of IOB chunks into single tokens."""

    if span_type == 'iobes':
        return to_spans_iobes(sequence, lut, verbose)

    strict_iob2 = (span_type == 'iob2') or (span_type == 'bio')
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]

        #if label.startswith('B-'):
        if not label.startswith('I-') and not label == 'O':
            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i ]

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2 and verbose:
                        print('Warning, type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (label, current[0], i))

                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2 and verbose:
                    print('Warning, unexpected format (I before B @ %d) %s' % (i, label))
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def to_spans_iobes(sequence, lut, verbose=False):
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]

        # This indicates a multi-word chunk start
        if label.startswith('B-'):

            # Flush existing chunk
            if current is not None:
                chunks.append('@'.join(current))
            # Create a new chunk
            current = [label.replace('B-', ''), '%d' % i]

        # This indicates a single word chunk
        elif label.startswith('S-'):

            # Flush existing chunk, and since this is self-contained, we will clear current
            if current is not None:
                chunks.append('@'.join(current))
                current = None

            base = label.replace('S-', '')
            # Write this right into the chunks since self-contained
            chunks.append('@'.join([base, '%d' % i]))

        # Indicates we are inside of a chunk already
        elif label.startswith('I-'):

            # This should always be the case!
            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if verbose:
                        print('Warning: I without matching previous B/I @ %d' % i)
                    current = [base, '%d' % i]

            else:
                if verbose:
                    print('Warning: I without a previous chunk @ %d' % i)
                current = [label.replace('I-', ''), '%d' % i]

        # We are at the end of a chunk, so flush current
        elif label.startswith('E-'):

            # Flush current chunk
            if current is not None:
                base = label.replace('E-', '')
                if base == current[0]:
                    current.append('%d' % i)
                    chunks.append('@'.join(current))
                    current = None
                else:
                    chunks.append('@'.join(current))
                    if verbose:
                        print('Warning: E doesnt agree with previous B/I type!')
                    current = [base, '%d' % i]
                    chunks.append('@'.join(current))
                    current = None

            # This should never happen
            else:
                current = [label.replace('E-', ''), '%d' % i]
                if verbose:
                    print('Warning, E without previous chunk! @ %d' % i)
                chunks.append('@'.join(current))
                current = None
        # Outside
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    # If something is left, flush
    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


@exporter
def f_score(overlap_count, gold_count, guess_count, f=1):
    beta_sq = f*f
    if guess_count == 0: return 0.0
    precision = overlap_count / float(guess_count)
    recall = overlap_count / float(gold_count)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    f = (1. + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return f


@exporter
def unzip_model(path):
    """If the path for a model file is a zip file, unzip it in /tmp and return the unzipped path"""
    if path.endswith("zip"):
        with open(path, 'rb') as f:
            sha1 = hashlib.sha1(f.read()).hexdigest()
        temp_dir = os.path.join("/tmp/", sha1)
        if not os.path.exists(temp_dir):
            print("unzipping model")
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
        if len(os.listdir(temp_dir)) == 1:  # a directory was zipped v files
            temp_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        path = os.path.join(temp_dir, [x[:-6] for x in os.listdir(temp_dir) if 'index' in x][0])
    return path


@exporter
def save_vectorizers(basedir, vectorizers, name='vectorizers'):
    import pickle
    save_md_file = '{}/{}-{}.pkl'.format(basedir, name, os.getpid())
    with open(save_md_file, 'wb') as f:
        pickle.dump(vectorizers, f)


@exporter
def save_vocabs(basedir, embeds_or_vocabs, name='vocabs'):
    for k, embeds_or_vocabs in embeds_or_vocabs.items():
        save_md = '{}/{}-{}-{}.json'.format(basedir, name, k, os.getpid())
        # Its a vocab
        if isinstance(embeds_or_vocabs, collections.Mapping):
            write_json(embeds_or_vocabs, save_md)
        # Type is embeds
        else:
            write_json(embeds_or_vocabs.vocab, save_md)


@exporter
def load_vocabs(directory):
    vocab_fnames = find_files_with_prefix(directory, 'vocabs')
    vocabs = {}
    for f in vocab_fnames:
        print(f)
        k = f.split('-')[-2]
        vocab = read_json(f)
        vocabs[k] = vocab
    return vocabs


@exporter
def load_vectorizers(directory):
    import pickle
    vectorizers_fname = find_files_with_prefix(directory, 'vectorizers')[0]
    with open(vectorizers_fname, "rb") as f:
        vectorizers = pickle.load(f)
    return vectorizers


@exporter
def unzip_files(zip_path):
    with open(zip_path, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
        temp_dir = os.path.join("/tmp/", sha1)
        if not os.path.exists(temp_dir):
            print("unzipping model")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
        if len(os.listdir(temp_dir)) == 1:  # a directory was zipped v files
            temp_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])
    return temp_dir


@exporter
def find_model_basename(directory):
    path = os.path.join(directory, [x for x in os.listdir(directory) if 'model' in x and '-md' not in x][0])
    print(path)
    path = path.split('.')[:-1]
    return '.'.join(path)


@exporter
def find_files_with_prefix(directory, prefix):
    return [os.path.join(directory, x) for x in os.listdir(directory) if x.startswith(prefix)]


@exporter
def zip_files(basedir):
    pid = str(os.getpid())
    tgt_zip_base = os.path.abspath(basedir)
    model_files = [x for x in os.listdir(basedir) if x.find(pid) >= 0 and os.path.isfile(os.path.join(basedir, x))]
    z = zipfile.ZipFile("{}-{}.zip".format(tgt_zip_base, pid), "w")
    for f in model_files:
        f = os.path.join(basedir, f)
        z.write(f)
        os.remove(f)
    z.close()


@exporter
def zip_model(path):
    """zips the model files"""
    print("zipping model files")
    model_files = [x for x in os.listdir(".") if path[2:] in x]
    z = zipfile.ZipFile("{}.zip".format(path), "w")
    for f in model_files:
        z.write(f)
        os.remove(f)
    z.close()


@exporter
def verbose_output(verbose, confusion_matrix):
    if verbose is None:
        return
    do_print = bool(verbose.get("console", False))
    outfile = verbose.get("file", None)
    if do_print:
        print(confusion_matrix)
    if outfile is not None:
        confusion_matrix.save(outfile)


@exporter
def get_env_gpus():
    return os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(',')


LESS_THAN_METRICS = {"avg_loss", "loss", "perplexity", "ppl"}


@exporter
def get_metric_cmp(metric, user_cmp=None, less_than_metrics=LESS_THAN_METRICS):
    if user_cmp is not None:
        return _try_user_cmp(user_cmp)
    if metric in less_than_metrics:
        return lt, six.MAXSIZE
    return gt, 0


def _try_user_cmp(user_cmp):
    user_cmp = user_cmp.lower()
    if user_cmp in {"lt", "less", "less than", "<", "less_than"}:
        return lt, six.MAXSIZE
    if user_cmp in {"le", "lte", "<="}:
        return le, six.MAXSIZE
    if user_cmp in {"ge", "gte", ">="}:
        return ge, 0
    return gt, 0


@exporter
def show_examples(model, es, rlut1, rlut2, vocab, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    batch_dict = es[si]

    lengths_key = model.src_lengths_key
    src_field = lengths_key.split('_')[0]
    src_array = batch_dict[src_field]
    if max_examples > 0:
        max_examples = min(max_examples, src_array.shape[0])

    for i in range(max_examples):
        example = {}
        # Batch first, so this gets a single example at once
        for k, v in batch_dict.items():
            example[k] = v[i, np.newaxis]

        print('========================================================================')
        sent = lookup_sentence(rlut1, example[src_field].squeeze(), reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, example['tgt'].squeeze())
        print('[Actual] %s' % sent)
        dst_i = model.predict(example)[0][0]
        sent = lookup_sentence(rlut2, dst_i)
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')


@exporter
def convert_seq2seq_golds(indices, lengths, rlut, subword_fix=lambda x: x):
    """Convert indices to words and format like a bleu reference corpus.

    :param indices: The indices of the gold sentence. Should be in the shape
        `[B, T]`. Iterating though axis=1 should yield ints.
    :param lengths: The length of the gold sentences.
    :param rlut: `dict[int] -> str` A lookup table from indices to words.

    :returns: List[List[List[str]]] Shape is [B, 1, T] where T is the number of
        words in that gold sentence
    """
    golds = []
    for idx, l in zip(indices, lengths):
        gold = idx[:l]
        gold_str = lookup_sentence(rlut, gold)
        gold = subword_fix(gold_str.split())
        golds.append([gold])
    return golds


@exporter
def convert_seq2seq_preds(indices, rlut, subword_fix=lambda x: x):
    """Convert indices to words and format like a bleu hypothesis corpus.

    :param indices: The indices of the predicted sentence. Should be in the
        shape `[B, T]`. Iterating though axis=1 should yield ints.
    :param rlut: `dict[int] -> str` A lookup table from indices to words.

    :returns: List[List[str]] Shape is [B, T] where T is the number of
        words in that predicted sentence
    """
    preds = []
    for idx in indices:
        pred_str = lookup_sentence(rlut, idx)
        pred = subword_fix(pred_str.split())
        preds.append(pred)
    return preds
