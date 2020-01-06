import six
from six.moves.urllib.request import urlretrieve
import os
import io
import re
import sys
import json
import pickle
import inspect
import hashlib
import logging
import zipfile
import platform
import importlib
from itertools import chain
from collections import Counter
from operator import lt, le, gt, ge
from contextlib import contextmanager
from functools import partial, update_wrapper, wraps
import numpy as np
import addons
import collections

__all__ = []
logger = logging.getLogger('baseline')
# These are inputs to models that shouldn't be saved out
MAGIC_VARS = ['sess', 'tgt', 'y', 'lengths', 'gpus']


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
def normalize_backend(name):
    allowed_backends = {'tf', 'pytorch', 'dy', 'keras'}
    name = name.lower()
    if name == 'tensorflow':
        name = 'tf'
    elif name == 'torch' or name == 'pyt':
        name = 'pytorch'
    elif name == 'dynet':
        name = 'dy'
    if name not in allowed_backends:
        raise ValueError("Supported backends are %s, got %s" % (allowed_backends, name))
    return name


@exporter
def get_logging_level(level):
    """Get the logging level as a logging module constant.

    :param level: `str` The log level to get.

    :returns: The log level, defaults to `INFO`
    """
    return getattr(logging, level.upper(), logging.INFO)


@exporter
def get_console_logger(name, level=None, env_key='LOG_LEVEL'):
    """A small default logging setup.

    This is a default logging setup to print json formatted logging to
    the console. This is used as a default for when baseline/mead is used
    as an API. This can be overridden with the logging config.

    The level defaults to `INFO` but can also be read from an env var
    of you choice with a back off to `LOG_LEVEL`

    :param name: `str` The logger to create.
    :param level: `str` The level to look for.
    :param env_key: `str` The env var to look in.

    :returns: logging.Logger
    """
    if level is None:
        level = os.getenv(env_key, os.getenv('LOG_LEVEL', 'INFO'))
    level = get_logging_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = JSONFormatter()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


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


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED
    }

    def format(self, record):
        if record.levelname in self.COLORS:
            return color(super(ColoredFormatter, self).format(record), self.COLORS[record.levelname])
        return super(ColoredFormatter, self).format(record)


class JSONFormatter(ColoredFormatter):
    """Format message as JSON if possible, log normally otherwise."""
    def format(self, record):
        try:
            if isinstance(record.msg, (list, dict)):
                return json.dumps(record.msg)
        except TypeError:
            pass
        return super(JSONFormatter, self).format(record)


class MakeFileHandler(logging.FileHandler):
    """A File logger that will create intermediate dirs if need be."""
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        log_dir = os.path.dirname(filename)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        super(MakeFileHandler, self).__init__(filename, mode, encoding, delay)


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
    if isinstance(v, bool): return v
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
    return isinstance(x, (collections.Sequence, collections.MappingView))


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
    """Read a YAML file in.  If no file is found and default value is set, return that instead.  Otherwise error

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
        from distutils.version import LooseVersion
        if LooseVersion(yaml.__version__) >= LooseVersion("5.1"):
            return yaml.load(f, Loader=yaml.FullLoader)
        return yaml.load(f)


@exporter
def read_config_file(config_file):
    """Read a config file. This method optionally supports YAML, if the dependency was already installed.  O.W. JSON plz

    :param config_file: (``str``) A path to a config file which should be a JSON file, or YAML if pyyaml is installed
    :return: (``dict``) An object
    """
    if config_file.endswith('.yml') or config_file.endswith('.yaml'):
        return read_yaml(config_file, strict=True)
    return read_json(config_file, strict=True)


@exporter
def validate_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


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
        logger.info("Reading config file '{}'".format(config_stream))
        return read_config_file(config_stream)
    config = config_stream
    if config_stream.startswith("$"):
        logger.info("Reading config from '{}'".format(config_stream))
        config = os.getenv(config_stream[1:])
    else:
        if validate_url(config_stream):
            path_to_save, _ = urlretrieve(config_stream)
            return read_config_stream(path_to_save)
        else:
            logger.info("No file found '{}...', loading as string".format(config_stream[:12]))
    return json.loads(config)


def write_sentence_conll(handle, sentence, gold, txt, idx2label):

    if len(txt) != len(sentence):
        txt = txt[:len(sentence)]

    try:
        for word, truth, guess in zip(txt, gold, sentence):
            handle.write('%s %s %s\n' % (word['text'], idx2label[truth], idx2label[guess]))
        handle.write('\n')
    except:
        logger.error('ERROR: Failed to write lines... closing file')
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


def _idempotent_append(element, data):
    """Append to a list if that element is not already in the list.

    :param element: The element to add to the list.
    :param data: `List` the list to add to.
    :returns: `List` the list with the element in it.
    """
    if element not in data:
        data.append(element)
    return data


def _parse_module_as_path(module_name):
    """Convert a path to a file to a format that it can be imported.

    :param module_name: The module as a path.
    :returns: `Tuple[str, str]` the module name (without a file ext) and the
        absolute path of the dir the file lives in (or '' if the module_name
        is just a filename).
    """
    module_dir, module_name = os.path.split(module_name)
    module_dir = os.path.realpath(os.path.expanduser(module_dir)) if module_dir else module_dir
    module_name, _ = os.path.splitext(module_name)
    return module_name, module_dir


@exporter
def import_user_module(module_name):
    """Load a module that is in the python path

    :param model_name: (``str``) - the name of the module
    :return:
    """
    addon_path = os.path.dirname(os.path.realpath(addons.__file__))
    _idempotent_append(addon_path, sys.path)
    if any(module_name.endswith(suffix) for suffix in importlib.machinery.SOURCE_SUFFIXES):
        module_path = module_name
        module_name, _ = _parse_module_as_path(module_path)
        # File based import from here https://docs.python.org/3.6/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Set this module in sys.modules so later we can import the module by name when pickling things.
        sys.modules[module_name] = mod
        return mod
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
    logger.info('model file [%s]' % name)
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
@optional_params
def str_file(func, **kwargs):
    """A decorator to automatically open arguments that are files.

    If there are kwargs then they are name=mode. When the function is
    called if the argument name is a string then the file is opened with
    mode.

    If there are no kwargs then it is assumed the first argument is a
    file that should be opened as 'r'
    """
    possible_files = kwargs
    # We need to have the generator check out here so that the inner function is
    # either a generator (has a yield) or not. If we were to try to have this if
    # inside of the open_files then we would always return a generator.
    if inspect.isgeneratorfunction(func):
        @six.wraps(func)
        def open_files(*args, **kwargs):
            # If no arg names are given then assume the first arg is a file
            # you want to read from.
            if not possible_files:
                if isinstance(args[0], six.string_types):
                    with io.open(args[0], mode='r', encoding='utf-8') as f:
                        # Call the function with the file instead we need to
                        # yield from it until it is done other wise the file
                        # will be closed after the first yield.
                        for x in func(f, *args[1:], **kwargs):
                            yield x
                else:
                    for x in func(*args, **kwargs):
                        yield x
            else:
                # Otherwise we have multiple files we want to open
                to_close = []
                # Get a dict representation of what it will look like if we
                # call func with *args and **kwargs
                arg = inspect.getcallargs(func, *args, **kwargs)
                try:
                    for f, mode in possible_files.items():
                        if isinstance(arg[f], six.string_types):
                            # Replace strings with the opened files
                            arg[f] = io.open(arg[f], mode=mode, encoding=None if 'b' in mode else 'utf-8')
                            to_close.append(f)
                    # Call the function with the files instead
                    for x in func(**arg):
                        yield x
                finally:
                    # Make sure to close the files
                    for f in to_close:
                        arg[f].close()
    else:
        @six.wraps(func)
        def open_files(*args, **kwargs):
            # If no arg names are given then assume the first arg is a file
            # you want to read from.
            if not possible_files:
                if isinstance(args[0], six.string_types):
                    with io.open(args[0], mode='r', encoding='utf-8') as f:
                        # Call the function with the file instead
                        return func(f, *args[1:], **kwargs)
                else:
                    return func(*args, **kwargs)
            else:
                # Otherwise we have multiple files we want to open
                to_close = []
                # Get a dict representation of what it will look like if we
                # call func with *args and **kwargs
                arg = inspect.getcallargs(func, *args, **kwargs)
                try:
                    for f, mode in possible_files.items():
                        if isinstance(arg[f], six.string_types):
                            # Replace strings with the opened files
                            arg[f] = io.open(arg[f], mode=mode, encoding=None if 'b' in mode else 'utf-8')
                            to_close.append(f)
                    # Call the function with the files instead
                    return func(**arg)
                finally:
                    # Make sure to close the files
                    for f in to_close:
                        arg[f].close()

    return open_files


@exporter
def normalize_indices(xs, length):
    """Normalize negative indices into positive.

    :param xs: `List[int]` The indices.
    :param length: `int` The length of the thing to be indexed

    :returns: `List[int]` The indices converted to positive only.
    """
    return list(map(lambda x: length + x if x < 0 else x, xs))


@exporter
def convert_iob_to_bio(seq):
    """Convert a sequence of IOB tags to BIO tags.

    The difference between IOB and BIO (also called IOB2) is that in IOB
    the B- prefix is only used to separate two chunks of the same type
    while in BIO the B- prefix is used to start every chunk.

    :param seq: `List[str]` The list of IOB tags.

    :returns: `List[str] The list of BIO tags.
    """
    new = []
    prev = 'O'
    for token in seq:
        # Only I- needs to be changed
        if token.startswith('I-'):
            # If last was O or last was a different type force B
            if prev == 'O' or token[2:] != prev[2:]:
                token = 'B-' + token[2:]
        new.append(token)
        prev = token
    return new


@exporter
def convert_bio_to_iob(seq):
    """Convert a sequence of BIO tags to IOB tags.

    The difference between BIO and IOB is that in IOB the B- prefix is only
    used to separate two chunks of the same type while in BIO the B- prefix
    starts every chunk. To convert we only need to look at the B- tokens.
    If they are following a chunk of the same type we leave it as a B-
    otherwise it converts it back to an I-

    :param seq: `List[str]` The list of BIO tags.

    :returns: `List[str]` The list of IOB tags.
    """
    new = []
    prev_ty = 'O'
    for token in seq:
        ty = "O" if token == "O" else token[2:]
        # In IOB, `B-` is only needed if the previous type is the same as ours
        if token.startswith('B-'):
            # If we are different than the type before us convert to `I-`
            if prev_ty != ty:
                token = 'I-' + ty
        new.append(token)
        prev_ty = ty
    return new


@exporter
def convert_bio_to_iobes(seq):
    """Convert a sequence of BIO tags to IOBES tags.

    The difference between BIO and IOBES tags is that in IOBES the end
    of a multi-token entity is marked with the E- prefix while in BIO
    it would end with an I- prefix.

    The other difference is that a single token entity in BIO is a
    just a B- whereas in IOBES it uses the special S- prefix.

    :param seq: `List[str]` The list of BIO tags.

    :returns: `List[str]` The list of IOBES tags.
    """
    new = []
    # Get tag bigrams with a fake O at the end so that the final real
    # token is processed correctly (Final I turned to E, etc.)
    for c, n in zip(seq, chain(seq[1:], ['O'])):
        if c.startswith('B-'):
            # If the next token is an I of the same class this is a
            # multi token span and should stay as B
            if n == c.replace('B-', 'I-'):
                new.append(c)
            # If the next token is anything else this is a single token
            # span and should become a S
            else:
                new.append(c.replace('B-', 'S-'))
        elif c.startswith('I-'):
            # If the next token is also an I of this type then we are
            # in the middle of a span and should stay I
            if n == c:
                new.append(c)
            # If next is anything else we are the end and thus E
            else:
                new.append(c.replace('I-', 'E-'))
        # Pass O through
        else:
            new.append(c)
    return new


@exporter
def convert_iobes_to_bio(seq):
    """Convert a sequence of IOBES tags to BIO tags

    :param seq: `List[str]` The list of IOBES tags.

    :returns: `List[str]` The list of BIO tags.
    """
    # Use re over `.replace` to make sure it only acts on the beginning of the token.
    return list(map(lambda x: re.sub(r'^S-', 'B-', re.sub(r'^E-', 'I-', x)), seq))


@exporter
def convert_iob_to_iobes(seq):
    """Convert a sequence of IOB tags to IOBES tags.

    :param seq: `List[str]` The list of IOB tags.

    :returns: `List[str]` The list of IOBES tags.
    """
    return convert_bio_to_iobes(convert_iob_to_bio(seq))


@exporter
def convert_iobes_to_iob(seq):
    """Convert a sequence of IOBES tags to IOB tags.

    :param seq: `List[str]` The list of IOBES tags.

    :returns: `List[str]` The list of IOB tags.
    """
    return convert_bio_to_iob(convert_iobes_to_bio(seq))


@str_file
def _sniff_conll_file(f, delim=None):
    """Figure out how many columns are in a conll file.

    :param file_name: `str` The name of the file.
    :param delim: `str` The token between columns in the file.

    :returns: `int` The number of columns in the file.
    """
    start = f.tell()
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        parts = line.split(delim)
        if len(parts) > 1:
            f.seek(start)
            return len(parts)


@exporter
@str_file
def read_conll(f, doc_pattern=None, delim=None, metadata=False, allow_comments=False, comment_pattern="#"):
    """Read from a conll file.

    :param f: `str` The file to read from.
    :param doc_pattern: `str` A pattern that matches the line that signals the
        beginning of a new document in the conll file. When None just the
        conll sentences are returned.
    :param delim: `str` The token between columns in the file
    :param metadata: `bool` Should meta data (lines starting with `#` before a
        sentence) be returned with our sentences.
    :param allow_comments: `bool` Are comments (lines starting with `#`) allowed in the file.

    :returns: `Generator` The sentences or documents from the file.
    """
    if metadata and not allow_comments:
        raise ValueError(
            "You have metadata set to `True` but allow_comments set to `False` you can't extract "
            "metadata from a file that doesn't allow comments."
        )
    if doc_pattern is not None:
        if metadata:
            for x in read_conll_docs_md(f, doc_pattern, delim=delim, comment_pattern=comment_pattern):
                yield x
        else:
            for x in read_conll_docs(f, doc_pattern, delim=delim, allow_comments=allow_comments, comment_pattern=comment_pattern):
                yield x
    else:
        if metadata:
            for x in read_conll_sentences_md(f, delim=delim, comment_pattern=comment_pattern):
                yield x
        else:
            for x in read_conll_sentences(f, delim=delim, allow_comments=allow_comments, comment_pattern=comment_pattern):
                yield x


@str_file
def read_conll_sentences(f, delim=None, allow_comments=True, comment_pattern="#"):
    """Read sentences from a conll file.

    :param f: `str` The file to read from.
    :param delim: `str` The token between columns in the file.

    Note:
        If you have a sentence where the first token is `#` it will get eaten by the
        metadata. If this happens you need to set `allow_comments=True` and not have
        comments in the file. If you have comments in the file and set this then
        they will show up in the sentences

    :returns: `Generator[List[List[str]]]` A list of rows representing a sentence.
    """
    sentence = []
    for line in f:
        line = line.rstrip()
        # Comments are not allowed in the middle of a sentence so if we find a line that
        # starts with # but we are in a sentence it must be a # as a token so we should
        # not skip it
        if allow_comments and not sentence and line.startswith(comment_pattern): continue
        # Blank lines signal the end of a sentence
        if len(line) == 0:
            # If we built a sentence yield it, this check allows multiple blank lines in a row
            if sentence:
                yield sentence
            # Reset the sentence
            sentence = []
            continue
        # This is a normal row, we split and take the tokens.
        sentence.append(line.split(delim))
    # If we have a sentence then the file didn't end with a new line, we yield the sentence
    # so we don't lose it
    if sentence:
        yield sentence


@str_file
def read_conll_sentences_md(f, delim=None, comment_pattern="#"):
    """Read sentences from a conll file.

    :param f: `str` The file to read from.
    :param delim: `str` The token between columns in the file.

    Note:
        If there are document annotations in the conll file then they will show
        up in the meta data for what would be the first sentence of that doc

        If you have a sentence where the first token is `#` it will show up in the
        metadata. If this happens you'll need to update you comments to use a different
        comment pattern, something like `# comment:` I recommend having a space in
        you patten so it can't show up as a conll token

    :returns: `Generator[Tuple[List[List[str]], List[List[str]]]`
        The first element is the list or rows, the second is a list of comment
        lines that preceded that sentence in the file.
    """
    sentence, meta = [], []
    for line in f:
        line = line.rstrip()
        # Comments are not allowed in the middle of a sentence so if we find a line that
        # starts with # but we are in a sentence it must be a # as a token so we should
        # not skip it. If this is a comment we track it in our meta data list
        if not sentence and line.startswith(comment_pattern):
            meta.append(line)
            continue
        if len(line) == 0:
            if sentence:
                yield sentence, meta
            sentence, meta = [], []
            continue
        sentence.append(line.split(delim))
    if sentence:
        yield sentence, meta


@str_file
def read_conll_docs(f, doc_pattern="# begin doc", delim=None, allow_comments=True, comment_pattern="#"):
    """Read sentences from a conll file.

    :param f: `str` The file to read from.
    :param doc_pattern: `str` The beginning of lines that represent new documents
    :param delim: `str` The token between columns in the file.

    Note:
        If you have a sentence where the first token is `#` it will show up in the
        metadata. If this happens you'll need to update you comments to use a different
        comment pattern, something like `# comment:` I recommend having a space in
        you patten so it can't show up as a conll token

    :returns: `Generator[List[List[List[str]]]]`
        A document which is a list of sentences.
    """
    doc, sentence = [], []
    for line in f:
        line = line.rstrip()
        if line.startswith(doc_pattern):
            if doc:
                if sentence:
                    doc.append(sentence)
                yield doc
                doc, sentence = [], []
            continue
        elif allow_comments and not sentence and line.startswith(comment_pattern):
            continue
        if len(line) == 0:
            if sentence:
                doc.append(sentence)
                sentence = []
            continue
        sentence.append(line.split(delim))
    if doc or sentence:
        if sentence:
            doc.append(sentence)
        yield doc


@str_file
def read_conll_docs_md(f, doc_pattern="# begin doc", delim=None, comment_pattern="#"):
    """Read sentences from a conll file.

    :param f: `str` The file to read from.
    :param doc_pattern: `str` The beginning of lines that represent new documents
    :param delim: `str` The token between columns in the file.

    Note:
        If you have a sentence where the first token is `#` it will show up in the
        metadata. If this happens you'll need to update you comments to use a different
        comment pattern, something like `# comment:` I recommend having a space in
        you patten so it can't show up as a conll token

    :returns: `Generator[Tuple[List[List[List[str]]], List[str]  List[List[str]]]`
        The first element is a document, the second is a list of comments
        lines that preceded the document break (includes the document line)
        since the last sentence. The last is a list of comments for each
        list in the document.
    """
    doc, sentence, doc_meta, sent_meta, meta = [], [], [], [], []
    for line in f:
        line = line.rstrip()
        if line.startswith(doc_pattern):
            new_doc_meta = meta
            meta = []
            new_doc_meta.append(line)
            if doc:
                if sentence:
                    doc.append(sentence)
                    sentence = []
                yield doc, doc_meta, sent_meta
                doc, sentence, sent_meta = [], [], []
            doc_meta = new_doc_meta
            continue
        elif not sentence and line.startswith(comment_pattern):
            meta.append(line)
            continue
        if len(line) == 0:
            if sentence:
                doc.append(sentence)
                sent_meta.append(meta)
                sentence, meta = [], []
            continue
        sentence.append(line.split(delim))
    if doc or sentence:
        if sentence:
            doc.append(sentence)
            sent_meta.append(meta)
            meta = []
        yield doc, doc_meta, sent_meta


@exporter
@str_file(ifile='r', ofile='w')
def convert_conll_file(ifile, ofile, convert, fields=[-1], delim=None):
    """Convert the tagging scheme in a conll file.

    This function assumes the that columns that one wishes to convert are
    the right model columns.

    :param ifile: `str` The input file name.
    :param ofile: `str` The output file name.
    :param convert: `Callable(List[str]) -> List[str]` The function that
        transforms a sequence in one tag scheme to another scheme.
    :param fields: `List[int]` The columns to convert.
    :param delim: `str` The symbol that separates the columns.
    """
    output_delim = ' ' if delim is None else delim
    conll_length = _sniff_conll_file(ifile, delim)
    fields = set(normalize_indices(fields, conll_length))
    for lines, md in read_conll_sentences_md(ifile, delim=delim):
        lines = zip(*(convert(l) if i in fields else l for i, l in enumerate(zip(*lines))))
        # Write out meta data
        if md:
            ofile.write('\n'.join(md) + '\n')
        # Write out the lines
        ofile.write('\n'.join(output_delim.join(l).rstrip() for l in lines) + '\n\n')


@exporter
@str_file(ifile='r', ofile='w')
def convert_iob_conll_to_bio(ifile, ofile, fields=[-1], delim=None):
    """Convert a conll file from iob to bio."""
    convert_conll_file(ifile, ofile, convert_iob_to_bio, fields, delim)


@exporter
@str_file(ifile='r', ofile='w')
def convert_iob_conll_to_iobes(ifile, ofile, fields=[-1], delim=None):
    """Convert a conll file from iob to iobes."""
    convert_conll_file(ifile, ofile, convert_iob_to_iobes, fields, delim)


@exporter
@str_file(ifile='r', ofile='w')
def convert_bio_conll_to_iobes(ifile, ofile, fields=[-1], delim=None):
    """Convert a conll file from bio to iobes."""
    convert_conll_file(ifile, ofile, convert_bio_to_iobes, fields, delim)


@exporter
@str_file(ifile='r', ofile='w')
def convert_iobes_conll_to_bio(ifile, ofile, fields=[-1], delim=None):
    """Convert a conll file from iobes to bio. Useful for formatting output to use `conlleval.pl`."""
    convert_conll_file(ifile, ofile, convert_iobes_to_bio, fields, delim)


@exporter
def to_spans(sequence, lut, span_type, verbose=False, delim="@"):
    """Turn a sequence into a list of chunks.

    :param sequence: `List[int]` The tag sequence.
    :param lut: `Dict[int] -> str` A mapping for integers to tag names.
    :param span_type: `str` The tagging scheme.
    :param verbose: `bool` Should we output warning on illegal transitions.
    :param delim: `str` The symbol the separates output chunks from their indices.

    :returns: `List[str]` The list of entities in the order they appear. The
        entities are in the form {chunk_type}{delim}{index}{delim}{index}...
        for example LOC@3@4@5 means a Location chunk was at indices 3, 4, and 5
        in the original sequence.
    """
    sequence = [lut[y] for y in sequence]
    return to_chunks(sequence, span_type, verbose, delim)


@exporter
def to_chunks(sequence, span_type, verbose=False, delim="@"):
    """Turn a sequence of tags into a list of chunks.

    :param sequence: `List[str]` The tag sequence.
    :param span_type: `str` The tagging scheme.
    :param verbose: `bool` Should we output warning on illegal transitions.
    :param delim: `str` The symbol the separates output chunks from their indices.

    :returns: `List[str]` The list of entities in the order they appear. The
        entities are in the form {chunk_type}{delim}{index}{delim}{index}...
        for example LOC@3@4@5 means a Location chunk was at indices 3, 4, and 5
        in the original sequence.
    """
    if span_type == 'iobes':
        return to_chunks_iobes(sequence, verbose, delim)

    strict_iob2 = (span_type == 'iob2') or (span_type == 'bio')
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None
    for i, label in enumerate(sequence):
        if not label.startswith('I-') and not label == 'O':
            if current is not None:
                chunks.append(delim.join(current))
            current = [label.replace('B-', ''), '%d' % i ]
        elif label.startswith('I-'):
            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append(delim.join(current))
                    if iobtype == 2 and verbose:
                        logger.warning('Warning: type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (label, current[0], i))

                    current = [base, '%d' % i]
            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2 and verbose:
                    logger.warning('Warning: unexpected format (I before B @ %d) %s' % (i, label))
        else:
            if current is not None:
                chunks.append(delim.join(current))
            current = None
    if current is not None:
        chunks.append(delim.join(current))
    return chunks


def to_chunks_iobes(sequence, verbose=False, delim="@"):
    """Turn a sequence of IOBES tags into a list of chunks.

    :param sequence: `List[str]` The tag sequence.
    :param verbose: `bool` Should we output warning on illegal transitions.
    :param delim: `str` The symbol the separates output chunks from their indices.

    :returns: `List[str]` The list of entities in the order they appear. The
        entities are in the form {chunk_type}{delim}{index}{delim}{index}...
        for example LOC@3@4@5 means a Location chunk was at indices 3, 4, and 5
        in the original sequence.
    """
    chunks = []
    current = None
    for i, label in enumerate(sequence):
        # This indicates a multi-word chunk start
        if label.startswith('B-'):
            # Flush existing chunk
            if current is not None:
                chunks.append(delim.join(current))
            # Create a new chunk
            current = [label.replace('B-', ''), '%d' % i]
            if verbose:
                # Look ahead to make sure this `B-Y` shouldn't be a `S-Y`
                if i < len(sequence) - 1 and label[2:] != sequence[i + 1][2:]:
                    logger.warning('Warning: Single B token chunk @ %d', i)
        # This indicates a single word chunk
        elif label.startswith('S-'):
            # Flush existing chunk, and since this is self-contained, we will clear current
            if current is not None:
                chunks.append(delim.join(current))
                current = None
            base = label.replace('S-', '')
            # Write this right into the chunks since self-contained
            chunks.append(delim.join([base, '%d' % i]))
        # Indicates we are inside of a chunk already
        elif label.startswith('I-'):
            # This should always be the case!
            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append(delim.join(current))
                    if verbose:
                        logger.warning('Warning: I without matching previous B/I @ %d' % i)
                    current = [base, '%d' % i]
            else:
                if verbose:
                    logger.warning('Warning: I without a previous chunk @ %d' % i)
                current = [label.replace('I-', ''), '%d' % i]
        # We are at the end of a chunk, so flush current
        elif label.startswith('E-'):
            # Flush current chunk
            if current is not None:
                base = label.replace('E-', '')
                if base == current[0]:
                    current.append('%d' % i)
                    chunks.append(delim.join(current))
                    current = None
                else:
                    chunks.append(delim.join(current))
                    if verbose:
                        logger.warning("Warning: E doesn't agree with previous B/I type!")
                    current = [base, '%d' % i]
                    chunks.append(delim.join(current))
                    current = None
            # This should never happen
            else:
                current = [label.replace('E-', ''), '%d' % i]
                if verbose:
                    logger.warning('Warning: E without previous chunk! @ %d' % i)
                chunks.append(delim.join(current))
                current = None
        # Outside
        else:
            if current is not None:
                chunks.append(delim.join(current))
            current = None
    # If something is left, flush
    if current is not None:
        chunks.append(delim.join(current))
    return chunks


@exporter
def span_f1(golds, preds):
    """Calculate Span level F1 score.

    :param golds: `List[set[str]]` The list of the set of gold chunks.
    :param preds: `List[set[str]]` The list of the set of predicted chunks.

    :returns: `float` The f1 score.
    """
    overlap = sum(len(g & p) for g, p in zip(golds, preds))
    gold_total = sum(len(g) for g in golds)
    pred_total = sum(len(p) for p in preds)
    return f_score(overlap, gold_total, pred_total)


@exporter
def per_entity_f1(golds, preds, delim="@"):
    """Calculate Span level F1 with break downs per entity type.

    :param golds: `List[set[str]]` The list of the set of gold chunks.
    :param preds: `List[set[str]]` The list of the set of predicted chunks.
    :param delim: `str` The symbol that separates an entity from its indices.

    :returns: `dict` The metrics at a global level and fine grained entity
        level performance.

    Note:
        This function returns most of the metrics needed for the
        `conlleval_output`. `acc` and `tokens` (the token level accuracy
        and the number of tokens respectively) need to be added.
    """
    metrics = {}
    overlap = Counter()
    gold_total = Counter()
    pred_total = Counter()
    types = set()
    for g, p in zip(golds, preds):
        overlaps = g & p
        overlap['total'] += len(overlaps)
        gold_total['total'] += len(g)
        pred_total['total'] += len(p)
        for o in overlaps:
            ent = o.split(delim)[0]
            overlap[ent] += 1
            types.add(ent)
        for o in g:
            ent = o.split(delim)[0]
            gold_total[ent] += 1
            types.add(ent)
        for o in p:
            ent = o.split(delim)[0]
            pred_total[ent] += 1
            types.add(ent)
    metrics['overlap'] = overlap['total']
    metrics['gold_total'] = gold_total['total']
    metrics['pred_total'] = pred_total['total']
    metrics['precision'] = precision(overlap['total'], pred_total['total']) * 100
    metrics['recall'] = recall(overlap['total'], gold_total['total']) * 100
    metrics['f1'] = f_score(overlap['total'], gold_total['total'], pred_total['total']) * 100
    metrics['types'] = []
    for t in sorted(types):
        metrics['types'].append({
            'ent': t,
            'precision': precision(overlap[t], pred_total[t]) * 100,
            'recall': recall(overlap[t], gold_total[t]) * 100,
            'f1': f_score(overlap[t], gold_total[t], pred_total[t]) * 100,
            'count': pred_total[t]
        })
    return metrics


@exporter
def conlleval_output(results):
    """Create conlleval formated output.

    :param results: `dict` The metrics. results should have the following keys.
        tokens: `int` The total number of tokens processed.
        acc: `float` The token level accuracy.
        gold_total: `int` The total number of gold entities.
        pred_total: `int` The total number of predicted entities.
        overlap: `int` The number of exact match entites.
        precision: `float` The precision of all entities.
        recall: `float` The recall of all entities.
        f1: `float` The f1 score of all entities.
        types: `List[dict]` A list of metrics for each entity type. Keys should include:
            ent: `str` The name of the entity.
            precision: `float` The precision of this entity type.
            recall: `float` The recall of this this entity type.
            f1: `float` The f1 score of this entity type.
            count: `int` The number of predicted entities of this type.

    :returns: `str` The formatted string ready for printing.

    Note:
        Both the metrics in the results dict and acc are expected to already be
        multiplied by 100. The result won't look correct and a lot of the
        metric will be cut off if they are not.

        Metrics per type are output in the order they appear in the list.
        conlleval.pl outputs the types in sorted order. To match this the list
        in `results['types'] should be sorted.
    """
    s = "processed {tokens} tokens with {gold_total} phrases; found: {pred_total} phrases; correct: {overlap}.\n" \
            "accuracy: {acc:>{length}.2f}%; precision: {precision:>6.2f}%; recall: {recall:>6.2f}%; FB1: {f1:>6.2f}\n"
    t = []
    longest_ent = max(len(max(results['types'], key=lambda x: len(x['ent']))['ent']), 17)
    for type_metric in results['types']:
        t.append("{ent:>{longest_ent}}: precision: {precision:>6.2f}%; recall: {recall:>6.2f}%; FB1: {f1:>6.2f}  {count}".format(longest_ent=longest_ent, **type_metric))
    s = s + "\n".join(t)
    s = s.format(length=longest_ent - 11, **results)
    return s


@exporter
def precision(overlap_count, guess_count):
    """Compute the precision in a zero safe way.

    :param overlap_count: `int` The number of true positives.
    :param guess_count: `int` The number of predicted positives (tp + fp)

    :returns: `float` The precision.
    """
    if guess_count == 0: return 0.0
    return overlap_count / float(guess_count)


@exporter
def recall(overlap_count, gold_count):
    """Compute the recall in a zero safe way.

    :param overlap_count: `int` The number of true positives.
    :param gold_count: `int` The number of gold positives (tp + fn)

    :returns: `float` The recall.
    """
    if gold_count == 0: return 0.0
    return overlap_count / float(gold_count)


@exporter
def f_score(overlap_count, gold_count, guess_count, f=1):
    """Compute the f1 score.

    :param overlap_count: `int` The number of true positives.
    :param gold_count: `int` The number of gold positives (tp + fn)
    :param guess_count: `int` The number of predicted positives (tp + fp)
    :param f: `int` The beta term to weight precision vs recall.

    :returns: `float` The f score
    """
    beta_sq = f*f
    if guess_count == 0: return 0.0
    p = precision(overlap_count, guess_count)
    r = recall(overlap_count, gold_count)
    if p == 0.0 or r == 0.0:
        return 0.0
    f = (1. + beta_sq) * (p * r) / (beta_sq * p + r)
    return f


@exporter
def unzip_model(path):
    """If the path for a model file is a zip file, unzip it in /tmp and return the unzipped path"""
    # Import inside function to avoid circular dep :(
    # TODO: future solution move the export code a different file so mime_type can import from it
    # rather then from here, this allows here to import mime_type
    if os.path.isdir(path):
        return path
    from baseline.mime_type import mime_type
    if mime_type(path) == 'application/zip':
        with open(path, 'rb') as f:
            sha1 = hashlib.sha1(f.read()).hexdigest()
        temp_dir = os.path.join("/tmp/", sha1)
        if not os.path.exists(temp_dir):
            logger.info("unzipping model")
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
        if len(os.listdir(temp_dir)) == 1:  # a directory was zipped v files
            temp_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        path = os.path.join(temp_dir, [x[:-6] for x in os.listdir(temp_dir) if 'index' in x][0])
    return path


@exporter
def save_vectorizers(basedir, vectorizers, name='vectorizers'):
    import pickle
    save_md_file = os.path.join(basedir, '{}-{}.pkl'.format(name, os.getpid()))
    with open(save_md_file, 'wb') as f:
        pickle.dump(vectorizers, f)
    # Save out the vectorizer module names so we can automatically import them
    # when reloading without going all the way to a pure json save
    vectorizer_modules = [v.__class__.__module__ for v in vectorizers.values()]
    module_file = os.path.join(basedir, '{}-{}.json'.format(name, os.getpid()))
    write_json(vectorizer_modules, module_file)


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
        logger.info(f)
        k = f.split('-')[-2]
        vocab = read_json(f)
        vocabs[k] = vocab
    return vocabs


@exporter
def load_vectorizers(directory):
    vectorizers_fname = find_files_with_prefix(directory, 'vectorizers')
    # Find the module list for the vectorizer so we can import them without
    # needing to bother the user with providing them
    vectorizers_modules = [x for x in vectorizers_fname if 'json' in x][0]
    modules = read_json(vectorizers_modules)
    for module in modules:
        import_user_module(module)
    vectorizers_pickle = [x for x in vectorizers_fname if 'pkl' in x][0]
    with open(vectorizers_pickle, "rb") as f:
        vectorizers = pickle.load(f)
    return vectorizers


@exporter
def unzip_files(zip_path):
    if os.path.isdir(zip_path):
        return zip_path
    from baseline.mime_type import mime_type
    if mime_type(zip_path) == 'application/zip':
        with open(zip_path, 'rb') as f:
            sha1 = hashlib.sha1(f.read()).hexdigest()
            temp_dir = os.path.join("/tmp/", sha1)
            if not os.path.exists(temp_dir):
                logger.info("unzipping model")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)
            if len(os.listdir(temp_dir)) == 1:  # a directory was zipped v files
                temp_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        return temp_dir
    return zip_path


@exporter
def find_model_basename(directory):
    path = os.path.join(directory, [x for x in os.listdir(directory) if 'model' in x and '-md' not in x][0])
    logger.info(path)
    path = path.split('.')[:-1]
    return '.'.join(path)


@exporter
def find_files_with_prefix(directory, prefix):
    return [os.path.join(directory, x) for x in os.listdir(directory) if x.startswith(prefix)]


@exporter
def zip_files(basedir):
    pid = str(os.getpid())
    tgt_zip_base = os.path.abspath(basedir)
    zip_name = os.path.basename(tgt_zip_base)
    model_files = [x for x in os.listdir(basedir) if pid in x and os.path.isfile(os.path.join(basedir, x))]
    with zipfile.ZipFile("{}-{}.zip".format(tgt_zip_base, pid), "w") as z:
        for f in model_files:
            abs_f = os.path.join(basedir, f)
            z.write(abs_f, os.path.join(zip_name, f))
            os.remove(abs_f)


@exporter
def zip_model(path):
    """zips the model files"""
    logger.info("zipping model files")
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
        logger.info(confusion_matrix)
    if outfile is not None:
        confusion_matrix.save(outfile)


@exporter
def get_env_gpus():
    return os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NVIDIA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0'))).split(',')


LESS_THAN_METRICS = {"avg_loss", "loss", "perplexity", "ppl"}


@exporter
def get_metric_cmp(metric, user_cmp=None, less_than_metrics=LESS_THAN_METRICS):
    if user_cmp is not None:
        return _try_user_cmp(user_cmp)
    if metric in less_than_metrics:
        return lt, six.MAXSIZE
    return gt, -six.MAXSIZE - 1


def _try_user_cmp(user_cmp):
    user_cmp = user_cmp.lower()
    if user_cmp in {"lt", "less", "less than", "<", "less_than"}:
        return lt, six.MAXSIZE
    if user_cmp in {"le", "lte", "<="}:
        return le, six.MAXSIZE
    if user_cmp in {"ge", "gte", ">="}:
        return ge, -six.MAXSIZE - 1
    return gt, -six.MAXSIZE - 1


@exporter
def show_examples(model, es, rlut1, rlut2, vocab, mxlen, sample, prob_clip, max_examples, reverse):
    """Expects model.predict to return [B, K, T]."""
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

        logger.info('========================================================================')
        sent = lookup_sentence(rlut1, example[src_field].squeeze(), reverse=reverse)
        logger.info('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, example['tgt'].squeeze())
        logger.info('[Actual] %s' % sent)
        dst_i = model.predict(example)[0][0]
        sent = lookup_sentence(rlut2, dst_i)
        logger.info('Guess: %s' % sent)
        logger.info('------------------------------------------------------------------------')


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
        gold = subword_fix(gold_str).split()
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
        pred = subword_fix(pred_str).split()
        preds.append(pred)
    return preds


@exporter
def undo_bpe(seq):
    """Undo the BPE splits to make Bleu comparable.

    :param seq: `str`: The string with encoded tokens in it.

    :returns: `str`: The string with BPE splits collapsed.
    """
    # BPE token is @@ this removes it if it is at the end of a word or the end
    # of the sequence.
    return re.sub(r"@@( | ?$)", "", seq)


@exporter
def undo_sentence_piece(seq):
    """Undo the sentence Piece splits to make Bleu comparable."""
    return seq.replace("\u2581", "")


@exporter
def ngrams(sentence, filtsz=3, joiner='@@'):
    """Generate ngrams over a sentence

    :param sentence: (`List[str]`) Some tokens
    :param filtsz: The ngram width
    :param joiner: A string to join ngrams
    :return: (`List[str]`) A list of ngrams
    """
    chunks = []
    nt = len(sentence)
    for i in range(nt - filtsz + 1):
        chunk = sentence[i:i+filtsz]
        chunks += [joiner.join(chunk)]
    return chunks


@exporter
@str_file
def read_label_first_data(f):
    """Read data from a file where the first token in the label and each line is a single example.

    :param f: `Union[str, IO]` The file to read from.
    :return: `Tuple[List[str], List[List[str]]]` The labels and text
    """
    labels, texts = [], []
    for line in f:
        line = line.rstrip()
        if line:
            label, text = line.split(maxsplit=1)
            labels.append(label)
            texts.append(text.split())
    return labels, texts


@exporter
@str_file(w='w')
def write_label_first_data(w, labels, texts):
    """Read data to a file where the first token in the label and each line is a single example.

    :param w: `Union[str, IO]` The file to write the results in
    :param labels: `List[str]` The labels for the examples
    :param texts: `List[List[str]]` The text examples
    """
    w.write("\n".join(" ".join(chain((l,), t)) for l, t in zip(labels, texts)))
