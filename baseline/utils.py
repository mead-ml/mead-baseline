import six
import os
import re
import sys
import json
import gzip
import pickle
import hashlib
import logging
import zipfile
import platform
import shutil
from functools import wraps
from operator import lt, le, gt, ge
from contextlib import contextmanager
from typing import Dict, List, Set, Optional
import numpy as np
import collections
import eight_mile
import importlib
from eight_mile.utils import *
from baseline.progress import create_progress_bar
import addons
from six.moves.urllib.request import urlretrieve

__all__ = []
__all__.extend(eight_mile.utils.__all__)
logger = logging.getLogger('baseline')
# These are inputs to models that shouldn't be saved out
MAGIC_VARS = ['sess', 'tgt', 'y', 'lengths', 'gpus']
MAGIC_VARS = ['sess', 'tgt', 'y', 'lengths']
MEAD_HUB_MODULES = []
DEFAULT_DATA_CACHE = os.path.expanduser('~/.bl-data')
export = exporter(__all__)


export(str2bool)

@export
@export
def import_user_module(module_name: str, data_download_cache: Optional[str] = None):
    """Load a module that is in the python path
    :param model_name: (``str``) - the name of the module
    :return:
    """
    if not data_download_cache and os.path.exists(DEFAULT_DATA_CACHE):
        data_download_cache = DEFAULT_DATA_CACHE
    if data_download_cache:
        if module_name.startswith("hub:") or module_name.startswith("http"):
            if module_name.startswith("hub:"):
                vec = module_name.split(":")
                version = vec[1]
                addons_literal = vec[2]
                rest = ":".join(vec[3:])
                if not rest.endswith(".py"):
                    rest += ".py"
                if addons_literal != "addons":
                    raise Exception("We only support downloading addons right now")
                module_name = f"http://raw.githubusercontent.com/mead-ml/hub/master/{version}/addons/{rest}"
                if module_name in MEAD_HUB_MODULES:
                    logger.warning(f"Skipping previously downloaded module: {module_name}")
                    return None
                MEAD_HUB_MODULES.append(module_name)
            module_name = AddonDownloader(module_name, data_download_cache, cache_ignore=True).download()

    # TODO: get rid of this!
    addon_path = os.path.dirname(os.path.realpath(addons.__file__))
    idempotent_append(addon_path, sys.path)
    if any(module_name.endswith(suffix) for suffix in importlib.machinery.SOURCE_SUFFIXES):
        module_path = module_name
        module_name, _ = parse_module_as_path(module_path)
        # File based import from here https://docs.python.org/3.6/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Set this module in sys.modules so later we can import the module by name when pickling things.
        sys.modules[module_name] = mod
        return mod
    mod = importlib.import_module(module_name)
    return mod


@export
def normalize_backend(name: str) -> str:
    allowed_backends = {'tf', 'pytorch'}
    name = name.lower()
    if name == 'tensorflow':
        name = 'tf'
    elif name == 'torch' or name == 'pyt':
        name = 'pytorch'
    if name not in allowed_backends:
        raise ValueError("Supported backends are %s, got %s" % (allowed_backends, name))
    return name


@export
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


@export
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull, redirect(sys.stdout, devnull), redirect(sys.stderr, devnull):
        yield


@export
class Colors(object):
    GREEN = '\033[32;1m'
    RED = '\033[31;1m'
    YELLOW = '\033[33;1m'
    BLACK = '\033[30;1m'
    CYAN = '\033[36;1m'
    RESTORE = '\033[0m'


@export
def color(msg: str, color: str) -> str:
    if platform.system() == 'Windows':
        return msg
    return f"{color}{msg}{Colors.RESTORE}"


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED
    }

    def format(self, record):
        if record.levelname in self.COLORS:
            return color(super(ColoredFormatter, self).format(record), self.COLORS[record.levelname])
        return super().format(record)


class JSONFormatter(ColoredFormatter):
    """Format message as JSON if possible, log normally otherwise."""
    def format(self, record):
        try:
            if isinstance(record.msg, (list, dict)):
                return json.dumps(record.msg)
        except TypeError:
            pass
        return super().format(record)


class MakeFileHandler(logging.FileHandler):
    """A File logger that will create intermediate dirs if need be."""
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        log_dir = os.path.dirname(filename)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        super().__init__(filename, mode, encoding, delay)


@export
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


@export
def web_cleanup(word):
    if word.startswith('http'): return 'URL'
    if word.startswith('@'): return '@@@@'
    if word.startswith('#'): return '####'
    if word == '"': return ','
    if word in UNREP_EMOTICONS: return ';)'
    if word == '<3': return '&lt;3'
    return word


@export
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


@export
def lookup_sentence(rlut: Dict[int, str], seq: List[str], reverse: bool = False, padchar: str = '') -> str:
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


@export
def topk(k, probs):
    """Get a sparse index (dictionary of top values)."""
    idx = np.argpartition(probs, probs.size-k)[-k:]
    sort = idx[np.argsort(probs[idx])][::-1]
    return dict(zip(sort, probs[sort]))


@export
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


@export
def unzip_model(path):
    path = unzip_files(path)
    return os.path.join(path, [x[:-6] for x in os.listdir(path) if 'index' in x][0])


@export
def save_vectorizers(basedir, vectorizers, name='vectorizers'):
    save_md_file = os.path.join(basedir, '{}-{}.pkl'.format(name, os.getpid()))
    with open(save_md_file, 'wb') as f:
        pickle.dump(vectorizers, f)
    # Save out the vectorizer module names so we can automatically import them
    # when reloading without going all the way to a pure json save
    vectorizer_modules = [v.__class__.__module__ for v in vectorizers.values()]
    module_file = os.path.join(basedir, '{}-{}.json'.format(name, os.getpid()))
    write_json(vectorizer_modules, module_file)


@export
def save_vocabs(basedir, embeds_or_vocabs, name='vocabs'):
    for k, embeds_or_vocabs in embeds_or_vocabs.items():
        save_md = '{}/{}-{}-{}.json'.format(basedir, name, k, os.getpid())
        # Its a vocab
        if isinstance(embeds_or_vocabs, collections.Mapping):
            write_json(embeds_or_vocabs, save_md)
        # Type is embeds
        else:
            write_json(embeds_or_vocabs.vocab, save_md)


@export
def load_vocabs(directory: str, suffix: Optional[str] = None):
    vocab_fnames = find_files_with_prefix(directory, 'vocabs', suffix)
    vocabs = {}
    for f in vocab_fnames:
        logger.info(f)
        k = f.split('-')[-2]
        vocab = read_json(f)
        vocabs[k] = vocab
    return vocabs


@export
def load_vectorizers(directory: str, data_download_cache: Optional[str] = None):
    vectorizers_fname = find_files_with_prefix(directory, 'vectorizers')
    # Find the module list for the vectorizer so we can import them without
    # needing to bother the user with providing them
    vectorizers_modules = [x for x in vectorizers_fname if 'json' in x][0]
    modules = read_json(vectorizers_modules)
    for module in modules:
        import_user_module(module, data_download_cache)
    vectorizers_pickle = [x for x in vectorizers_fname if 'pkl' in x][0]
    with open(vectorizers_pickle, "rb") as f:
        vectorizers = pickle.load(f)
    return vectorizers


@export
def unzip_files(zip_path):
    if os.path.isdir(zip_path):
        return zip_path
    from eight_mile.utils import mime_type
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


@export
def find_model_basename(directory, basename=None):
    if not basename:
        basename = [x for x in os.listdir(directory) if 'model' in x and '-md' not in x and 'wgt' not in x and '.assets' not in x][0]
    else:
        globname = os.path.join(directory, basename)
        if not os.path.isfile(globname):
            import glob
            out = glob.glob(f'{globname}*')
            out = [x for x in out if 'model' in x and '-md' not in x and 'wgt' not in x and '.assets' not in x][0]
            basename = out
    path = os.path.join(directory, basename)
    logger.info(path)
    path = path.split('.')[:-1]
    return '.'.join(path)


@export
def find_files_with_prefix(directory, prefix, suffix=None):

    files_with_prefix = [os.path.join(directory, x) for x in os.listdir(directory) if x.startswith(prefix)]
    if suffix:
        files_with_prefix = [f for f in files_with_prefix if f.endswith(suffix)]
    return files_with_prefix


@export
def zip_files(basedir, limit_to_pid=True):
    pid = str(os.getpid())
    tgt_zip_base = os.path.abspath(basedir)
    zip_name = os.path.basename(tgt_zip_base)
    if limit_to_pid:
        model_files = [x for x in os.listdir(basedir) if pid in x and os.path.isfile(os.path.join(basedir, x))]
    else:
        model_files = [x for x in os.listdir(basedir) if os.path.isfile(os.path.join(basedir, x))]
    with zipfile.ZipFile("{}-{}.zip".format(tgt_zip_base, pid), "w") as z:
        for f in model_files:
            abs_f = os.path.join(basedir, f)
            z.write(abs_f, os.path.join(zip_name, f))
            os.remove(abs_f)


@export
def zip_model(path):
    """zips the model files"""
    logger.info("zipping model files")
    model_files = [x for x in os.listdir(".") if path[2:] in x]
    z = zipfile.ZipFile("{}.zip".format(path), "w")
    for f in model_files:
        z.write(f)
        os.remove(f)
    z.close()


@export
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


@export
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


@export
def show_examples(model, es, rlut1, rlut2, vocab, mxlen, sample, prob_clip, max_examples, reverse):
    """Expects model.predict to return [B, K, T]."""
    try:
        si = np.random.randint(0, len(es))
        batch_dict = es[si]
    except:
        batch_dict = next(iter(es))

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


@export
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


@export
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


@export
def undo_bpe(seq: str) -> str:
    """Undo the BPE splits to make Bleu comparable.

    :param seq: The string with encoded tokens in it.

    :returns: The string with BPE splits collapsed.
    """
    # BPE token is @@ this removes it if it is at the end of a word or the end
    # of the sequence.
    return re.sub(r"@@( | ?$)", "", seq)

@export
def undo_wordpiece(seq: str) -> str:
    """Undo the WordPiece splits to make Bleu comparable.  Use BERT-style detok
    :param seq: The string with encoded tokens in it.

    :returns: The string with BPE splits collapsed.
    """
    return re.sub(r"\s+##", "", seq)

@export
def undo_sentence_piece(seq):
    """Undo the sentence Piece splits to make Bleu comparable.
    TODO: in what context does this actually work?  it doesnt do replacement as above
    """

    return seq.replace("\u2581", "")

DATA_CACHE_CONF = "data-cache.json"

@export
def delete_old_copy(file_name):
    if os.path.exists(file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)
        else:
            shutil.rmtree(file_name)
    return file_name


@export
def extract_gzip(file_loc):
    temp_file = delete_old_copy("{}.1".format(file_loc))
    with gzip.open(file_loc, 'rb') as f_in:
        with open(temp_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if mime_type(temp_file) == "application/x-tar":
        return extract_tar(temp_file)
    else:
        shutil.move(temp_file, file_loc)
        return file_loc


@export
def extract_tar(file_loc):
    import tarfile
    temp_file = delete_old_copy("{}.1".format(file_loc))
    with tarfile.open(file_loc, "r") as tar_ref:
        tar_ref.extractall(temp_file)
    if len(os.listdir(temp_file)) != 1:
        raise RuntimeError("tar extraction unsuccessful")
    return os.path.join(temp_file, os.listdir(temp_file)[0])


@export
def extract_zip(file_loc):
    temp_file = delete_old_copy("{}.1".format(file_loc))
    with zipfile.ZipFile(file_loc, "r") as zip_ref:
        zip_ref.extractall(temp_file)
    return temp_file


@export
def extractor(filepath, cache_dir, extractor_func):
    with open(filepath, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    logger.info("extracting file..")
    path_to_save = filepath if extractor_func is None else extractor_func(filepath)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    path_to_save_sha1 = os.path.join(cache_dir, sha1)
    delete_old_copy(path_to_save_sha1)
    shutil.move(path_to_save, path_to_save_sha1)
    logger.info("downloaded data saved in {}".format(path_to_save_sha1))
    return path_to_save_sha1


@export
def web_downloader(url, path_to_save=None):
    # Use a class to simulate the nonlocal keyword in 2.7
    class Context: pg = None
    def _report_hook(count, block_size, total_size):
        if Context.pg is None:
            length = int((total_size + block_size - 1) / float(block_size)) if total_size != -1 else 1
            Context.pg = create_progress_bar(length)
        Context.pg.update()

    if not path_to_save:
        path_to_save = "/tmp/data.dload-{}".format(os.getpid())
    try:
        path_to_save, _ = urlretrieve(url, path_to_save, reporthook=_report_hook)
        Context.pg.done()
    except Exception as e:  # this is too broad but there are too many exceptions to handle separately
        raise RuntimeError("failed to download data from [url]: {} [to]: {}".format(url, path_to_save))
    return path_to_save


@export
def update_cache(key, data_download_cache):
    dcache = read_json(os.path.join(data_download_cache, DATA_CACHE_CONF))
    if key not in dcache:
        return
    del dcache[key]
    write_json(dcache, os.path.join(data_download_cache, DATA_CACHE_CONF))


def _verify_file(file_loc):
    # dropbox doesn't give 404 in case the file does not exist, produces an HTML. The actual files are never HTMLs.
    if not os.path.exists(file_loc):
        return False

    if os.path.isfile(file_loc) and mime_type(file_loc) == "text/html":
        return False

    return True


@export
def is_file_correct(file_loc, data_dcache=None, key=None):
    """check if the file location mentioned in the json file is correct, i.e.,
    exists and not corrupted. This is needed when the direct download link/ path for a file
    changes and the user is unaware. This is not tracked by sha1 either. If it returns False, delete the corrupted file.
    Additionally, if the file location is a URL, i.e. exists in the cache, delete it so that it can be re-downloaded.

    Keyword arguments:
    file_loc -- location of the file
    data_dcache -- data download cache location (default None, for local system file paths)
    key -- URL for download (default None, for local system file paths)
    """
    if _verify_file(file_loc):
        return True
    # Some files are prefixes (the datasset.json has `train` and the data has `train.fr` and `train.en`)
    dir_name = os.path.dirname(file_loc)
    # When we are using this for checking embeddings file_loc is a url so we need this check.
    if os.path.exists(dir_name):
        files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.join(dir_name, f).startswith(file_loc)]
        if files and all(_verify_file(f) for f in files):
            return True
    delete_old_copy(file_loc)
    if key is not None:  # cache file validation
        update_cache(key, data_dcache)
    return False


@export
def is_dir_correct(dir_loc, dataset_desc, data_dcache, key, ignore_file_check=False):
    """check if the directory extracted from the zip location mentioned in the datasets json file is correct, i.e.,
    all files inside exist and are not corrupted. If not, we will update the cache try to re-download them.

    Keyword arguments:
    dir_loc -- location of the directory
    dataset_desc -- to know the individual file locations inside the directory
    data_dcache -- data download cache location
    key -- URL for download
    ignore_file_check --to handle enc_dec datasets, see later.
    """

    if not os.path.exists(dir_loc) or not os.path.isdir(dir_loc):
        update_cache(key, data_dcache)
        return False
    if ignore_file_check:  # for enc_dec tasks there's no direct downloads
        return True
    files = [os.path.join(dir_loc, dataset_desc[k]) for k in dataset_desc if k.endswith("_file")]
    for f in files:
        if not is_file_correct(f, key, data_dcache):
            return False
    return True


@export
class Downloader:
    ZIPD = {'application/gzip': extract_gzip, 'application/zip': extract_zip}

    def __init__(self, data_download_cache, cache_ignore):
        super().__init__()
        self.cache_ignore = cache_ignore
        self.data_download_cache = data_download_cache

    def download(self):
        pass


@export
class SingleFileDownloader(Downloader):
    def __init__(self, dataset_file, data_download_cache, cache_ignore=False):
        super().__init__(data_download_cache, cache_ignore)
        self.dataset_file = dataset_file
        self.data_download_cache = data_download_cache

    def download(self):
        file_loc = self.dataset_file
        if is_file_correct(file_loc):
            return file_loc
        elif validate_url(file_loc):  # is it a web URL? check if exists in cache
            url = file_loc
            dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
            dcache = read_json(dcache_path)
            if url in dcache and is_file_correct(dcache[url], self.data_download_cache, url) and not self.cache_ignore:
                logger.info("file for {} found in cache, not downloading".format(url))
                return dcache[url]
            else:  # download the file in the cache, update the json
                cache_dir = self.data_download_cache
                logger.info("using {} as data/embeddings cache".format(cache_dir))
                temp_file = web_downloader(url)
                dload_file = extractor(filepath=temp_file, cache_dir=cache_dir,
                                       extractor_func=Downloader.ZIPD.get(mime_type(temp_file), None))
                dcache.update({url: dload_file})
                write_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                return dload_file
        raise RuntimeError("the file [{}] is not in cache and can not be downloaded".format(file_loc))


@export
class AddonDownloader(Downloader):
    ADDON_SUBPATH = 'addons'
    """Grab addons and write them to the download cache
    """
    def __init__(self, dataset_file, data_download_cache, cache_ignore=False):
        super().__init__(data_download_cache, cache_ignore)
        self.dataset_file = dataset_file
        self.data_download_cache = data_download_cache

    def download(self):
        file_loc = self.dataset_file
        if is_file_correct(file_loc):
            return file_loc
        elif validate_url(file_loc):  # is it a web URL? check if exists in cache
            url = file_loc
            dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
            dcache = read_json(dcache_path)
            # If the file already exists in the cache
            if url in dcache and is_file_correct(dcache[url], self.data_download_cache, url) and not self.cache_ignore:
                logger.info("file for {} found in cache, not downloading".format(url))
                return dcache[url]
            # Otherwise, we want it to be placed in ~/.bl-cache/addons
            else:  # download the file in the cache, update the json
                cache_dir = self.data_download_cache
                addon_path = os.path.join(cache_dir, AddonDownloader.ADDON_SUBPATH)
                if not os.path.exists(addon_path):
                    os.makedirs(addon_path)
                path_to_save = os.path.join(addon_path, os.path.basename(file_loc))
                logger.info("using {} as data/addons cache".format(cache_dir))
                web_downloader(url, path_to_save)
                dcache.update({url: path_to_save})
                write_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                return path_to_save
        raise RuntimeError("the file [{}] is not in cache and can not be downloaded".format(file_loc))


@export
class DataDownloader(Downloader):
    def __init__(self, dataset_desc, data_download_cache, enc_dec=False, cache_ignore=False):
        super().__init__(data_download_cache, cache_ignore)
        self.dataset_desc = dataset_desc
        self.data_download_cache = data_download_cache
        self.enc_dec = enc_dec

    def download(self):
        dload_bundle = self.dataset_desc.get("download", None)
        if dload_bundle is not None:  # download a zip/tar/tar.gz directory, look for train, dev test files inside that.
            dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
            dcache = read_json(dcache_path)
            if dload_bundle in dcache and \
                    is_dir_correct(dcache[dload_bundle], self.dataset_desc, self.data_download_cache, dload_bundle,
                                   self.enc_dec) and not self.cache_ignore:
                download_dir = dcache[dload_bundle]
                logger.info("files for {} found in cache, not downloading".format(dload_bundle))
                return {k: os.path.join(download_dir, self.dataset_desc[k]) for k in self.dataset_desc
                        if k.endswith("_file")}
            else:  # try to download the bundle and unzip
                if not validate_url(dload_bundle):
                    raise RuntimeError("can not download from the given url")
                else:
                    cache_dir = self.data_download_cache
                    temp_file = web_downloader(dload_bundle)

                    download_dir = extractor(filepath=temp_file, cache_dir=cache_dir,
                                             extractor_func=Downloader.ZIPD.get(mime_type(temp_file), None))
                    if "sha1" in self.dataset_desc:
                        if os.path.split(download_dir)[-1] != self.dataset_desc["sha1"]:
                            raise RuntimeError("The sha1 of the downloaded file does not match with the provided one")
                    dcache.update({dload_bundle: download_dir})
                    write_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                    return {k: os.path.join(download_dir, self.dataset_desc[k]) for k in self.dataset_desc
                            if k.endswith("_file")}
        else:  # we have download links to every file or they exist
            if not self.enc_dec:
                return {k: SingleFileDownloader(self.dataset_desc[k], self.data_download_cache).download()
                        for k in self.dataset_desc if k.endswith("_file") and self.dataset_desc[k]}
            else:
                return {k: self.dataset_desc[k] for k in self.dataset_desc if k.endswith("_file")}
                # these files can not be downloaded because there's a post processing on them.


@export
class EmbeddingDownloader(Downloader):
    def __init__(self, embedding_file, embedding_dsz, embedding_sha1, data_download_cache, cache_ignore=False, unzip_file=True):
        super().__init__(data_download_cache, cache_ignore)
        self.embedding_file = embedding_file
        self.embedding_key = embedding_dsz
        self.data_download_cache = data_download_cache
        self.unzip_file = unzip_file
        self.sha1 = embedding_sha1

    @staticmethod
    def _get_embedding_file(loc, key):
        if os.path.isfile(loc):
            logger.info("embedding file location: {}".format(loc))
            return loc
        else:  # This is a directory, return the actual file
            files = [x for x in os.listdir(loc) if str(key) in x]
            if len(files) == 0:
                raise RuntimeError("No embedding file found for the given key [{}]".format(key))
            elif len(files) > 1:
                logger.info("multiple embedding files found for the given key [{}], choosing {}".format(key, files[0]))
            embed_file_loc = os.path.join(loc, files[0])
            return embed_file_loc

    def download(self):
        if is_file_correct(self.embedding_file):
            logger.info("embedding file location: {}".format(self.embedding_file))
            return self.embedding_file
        dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
        dcache = read_json(dcache_path)
        if self.embedding_file in dcache and not self.cache_ignore:
            download_loc = dcache[self.embedding_file]
            logger.info("files for {} found in cache".format(self.embedding_file))
            return self._get_embedding_file(download_loc, self.embedding_key)
        else:  # try to download the bundle and unzip
            url = self.embedding_file
            if not validate_url(url):
                raise RuntimeError("can not download from the given url")
            else:
                cache_dir = self.data_download_cache
                temp_file = web_downloader(url)
                unzip_fn = Downloader.ZIPD.get(mime_type(temp_file)) if self.unzip_file else None
                download_loc = extractor(filepath=temp_file, cache_dir=cache_dir,
                                         extractor_func=unzip_fn)
                if self.sha1 is not None:
                    if os.path.split(download_loc)[-1] != self.sha1:
                        raise RuntimeError("The sha1 of the downloaded file does not match with the provided one")
                dcache.update({url: download_loc})
                write_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                return self._get_embedding_file(download_loc, self.embedding_key)
