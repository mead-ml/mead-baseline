import six
import os
import re
import sys
import json
import pickle
import hashlib
import logging
import zipfile
import platform
from operator import lt, le, gt, ge
from contextlib import contextmanager
import numpy as np
import collections
#from eight_mile.utils import export, optional_params
from eight_mile.utils import *
from functools import partial, update_wrapper, wraps

__all__ = []
logger = logging.getLogger('baseline')
# These are inputs to models that shouldn't be saved out
MAGIC_VARS = ['sess', 'tgt', 'y', 'lengths', 'gpus']
MAGIC_VARS = ['sess', 'tgt', 'y', 'lengths']


exporter = export(__all__)


@exporter
def normalize_backend(name):
    allowed_backends = {'tf', 'pytorch'}
    name = name.lower()
    if name == 'tensorflow':
        name = 'tf'
    elif name == 'torch' or name == 'pyt':
        name = 'pytorch'
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
def unzip_model(path):
    """If the path for a model file is a zip file, unzip it in /tmp and return the unzipped path"""
    # Import inside function to avoid circular dep :(
    # TODO: future solution move the export code a different file so mime_type can import from it
    # rather then from here, this allows here to import mime_type
    if os.path.isdir(path):
        return path
    from eight_mile.mime_type import mime_type
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
    from eight_mile.mime_type import mime_type
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

