import os
import sys
import importlib
from functools import partial, update_wrapper, wraps
import numpy as np
import addons
import json

__all__ = []

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
def listify(x):
    """Take a scalar or list and make it a list
    
    :param x: The input to convert
    :return: A list
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    if x is None:
        return []
    return [x]

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


@exporter
def read_json(filepath, default_value={}):
    """Read a JSON file in.  If no file is found and default value is set, return that instead.  Otherwise error

    :param filepath: A file to load
    :param default_value: If the file doesnt exist, an alternate object to return, or if None, throw FileNotFoundError
    :return: A JSON object
    """
    if not os.path.exists(filepath):
        if default_value is None:
            raise FileNotFoundError('No file [] found'.format(filepath))
        return default_value
    with open(filepath) as f:
        return json.load(f)


@exporter
def read_config_file(config_file):
    """Read a config file. This method optionally supports YAML, if the dependency was already installed.  O.W. JSON plz

    :param config_file: (``str``) A path to a config file which should be a JSON file, or YAML if pyyaml is installed
    :return: (``dict``) An object
    """
    with open(config_file) as f:
        if config_file.endswith('.yml'):
            import yaml
            return yaml.load(f)
        return json.load(f)


@exporter
def write_json(content, filepath):
    with open(filepath, "w") as f:
        json.dump(content, f, indent=True)

@exporter
def import_user_module(module_type, model_type):
    """Load a module that is in the python path with a canonical name

    This method loads a user-defined model, which must exist in the `PYTHONPATH` and must also
    follow a fixed naming convention of `{module_type}_{model_type}.py`.  The module is dynamically
    loaded, at which point its creator or loader function should be called to instantiate the model.
    This is essentially a plugin, but its implementation is trivial.

    :param module_type: one of `classifier`, `tagger`, `seq2seq`, `lang`
    :param model_type: A name for the model, which is the suffix
    :return:
    """
    sys.path.append(os.path.dirname(os.path.realpath(addons.__file__)))
    module_name = "%s_%s" % (module_type, model_type)
    print('Loading user model %s' % module_name)
    mod = importlib.import_module(module_name)
    return mod

@exporter
def create_user_model(input_, output_, **kwargs):
    """Create a user-defined model

    This creates a model defined by the user.
    It first imports a module that must exist in the `PYTHONPATH`, with a named defined as
    `{task_type}_{model_type}.py`.  Once created, this user-defined model can be trained within
    the existing training programs

    :param input_: Some type of word vectors for the input
    :param output_: Things passed dealing with the output
    :param kwargs:
    :return: A user-defined model
    """
    model_type = kwargs['model_type']
    mod = import_user_module(kwargs['task_type'], model_type)
    return mod.create_model(input_, output_, **kwargs)

def wrapped_partial(func, name=None, *args, **kwargs):
    """
    When we use `functools.partial` the `__name__` is not defined which breaks
    our export function so we use update wrapper to give it a `__name__`.

    :param name: A new name that is assigned to `__name__` so that the name
    of the partial can be different than the wrapped function.
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    if name is not None:
        partial_func.__name__ = name
    return partial_func

create_user_classifier_model = exporter(
    wrapped_partial(
        create_user_model,
        task_type='classify',
        name='create_user_classifier_model'
    )
)
create_user_tagger_model = exporter(
    wrapped_partial(
        create_user_model,
        task_type='tagger',
        name='create_user_tagger_model'
    )
)
create_user_seq2seq_model = exporter(
    wrapped_partial(
        create_user_model,
        task_type='seq2seq',
        name='create_user_seq2seq_model'
    )
)


@exporter
def create_user_lang_model(embeddings, **kwargs):
    model_type = kwargs['model_type']
    mod = import_user_module('lang', model_type)
    return mod.create_model(embeddings, **kwargs)


@exporter
def create_user_trainer(model, **kwargs):
    """Create a user-defined trainer

    Given a model, create a custom trainer that will train the model.  This requires that the trainer
    module lives in the `PYTHONPATH`, and is named `trainer_{trainer_type}`.  Once instantiated, this trainer
    can be used by the `fit()` function within each task type

    :param model: The model to train
    :param kwargs:
    :return: A user-defined trainer
    """
    model_type = kwargs['trainer_type']
    mod = import_user_module("trainer", model_type)
    return mod.create_trainer(model, **kwargs)


@exporter
def load_user_model(outname, **kwargs):
    """Loads a user-defined model

    This loads a previously serialized model defined by the user.
    It first imports a module that must exist in the `PYTHONPATH`, with a named defined as
    `{task_type}_{model_type}.py`.  Once loaded, this user-defined model can be used within the driver programs

    :param outname: The name of the file where the model is serialized
    :param kwargs:
    :return: A user-defined model
    """
    model_type = kwargs['model_type']
    mod = import_user_module(kwargs['task_type'], model_type)
    return mod.load_model(outname, **kwargs)


load_user_classifier_model = exporter(
    wrapped_partial(
        load_user_model,
        task_type='classify',
        name='load_user_classifier_model'
    )
)
load_user_tagger_model = exporter(
    wrapped_partial(
        load_user_model,
        task_type='tagger',
        name='load_user_tagger_model'
    )
)
load_user_seq2seq_model = exporter(
    wrapped_partial(
        load_user_model,
        task_type='seq2seq',
        name='load_user_seq2seq_model'
    )
)
load_user_lang_model = exporter(
    wrapped_partial(
        load_user_model,
        task_type='lm',
        name='load_user_lang_model'
    )
)


@exporter
def get_model_file(dictionary, task, platform):
    """Model name file helper to abstract different DL platforms (FWs)

    :param dictionary:
    :param task:
    :param platform:
    :return:
    """
    base = dictionary.get('outfile', './%s-model' % task)
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
    return (' '.join([rlut[idx] if rlut[idx] != '<PAD>' else padchar for idx in s])).strip()


@exporter
def topk(k, probs):
    """Get a sparse index (dictionary of top values).

    Note:
        mutates input for efficiency
    """

    lut = {}
    i = 0

    while i < k:
        idx = np.argmax(probs)
        lut[idx] = probs[idx]
        probs[idx] = 0
        i += 1
    return lut

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
def seq_fill_y(nc, yidx):
    """Convert a `BxT` sparse array to a dense one, to expand labels 
    
    :param nc: (``int``) The number of labels
    :param yidx: The sparse array of the labels
    :return: A dense array
    """
    batchsz = yidx.shape[0]
    siglen = yidx.shape[1]
    dense = np.zeros((batchsz, siglen, nc), dtype=np.int)
    for i in range(batchsz):
        for j in range(siglen):
            idx = int(yidx[i, j])
            if idx > 0:
                dense[i, j, idx] = 1

    return dense


@exporter
def to_spans(sequence, lut, strict_iob2=False):
    """Turn a sequence of IOB chunks into single tokens."""

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
                    if iobtype == 2:
                        print('Warning, type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (label, current[0], i))

                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning, unexpected format (I before B @ %d) %s' % (i, label))
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

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
def tagger_featurizer(tagger, tokens, mxlen, maxw, zero_alloc, word_trans_fn, vocab_keys):
    xs = zero_alloc((1, mxlen), dtype=int)
    xs_ch = zero_alloc((1, mxlen, maxw), dtype=int)
    lengths = zero_alloc(1, dtype=int)
    lengths[0] = min(len(tokens), mxlen)
    data = {}
    if not type(tokens[0]) is list:  # support the existing case
        tokens = [[token] for token in tokens]
    for j in range(mxlen):
        if j == len(tokens):
            break
        token_features = tokens[j]
        if 'word' in vocab_keys:
            word_index = vocab_keys['word']
            words_vocab = tagger.get_vocab(vocab_type='word')
            w = token_features[word_index]
            xs[0, j] = words_vocab.get(word_trans_fn(w), 0)
            if 'char' in vocab_keys:
                nch = min(len(w), maxw)
                for k in range(nch):
                    chars_vocab = tagger.get_vocab(vocab_type='char')
                    xs_ch[0, j, k] = chars_vocab.get(w[k], 0)
        for key in vocab_keys:
            if not key == 'word' and not key == 'char':
                feature_index = vocab_keys[key]
                feature = token_features[feature_index]
                feature_vocab = tagger.get_vocab(vocab_type=key)
                data[key] = zero_alloc((1, mxlen), dtype=np.int)
                data[key][0, j] = feature_vocab[feature]
    data.update({'x': xs, 'xch': xs_ch, 'lengths': lengths})
    return data


@exporter
def tagger_featurizer_elmo(tagger, tokens, mxlen, maxw, zero_alloc, word_trans_fn, vocab_keys):
    xs = zero_alloc((1, mxlen), dtype=int)
    xs_lc = zero_alloc((1, mxlen), dtype=int)
    xs_ch = zero_alloc((1, mxlen, maxw), dtype=int)
    lengths = zero_alloc(1, dtype=int)
    lengths[0] = min(len(tokens), mxlen)
    data = {}
    if not type(tokens[0]) is list:  # support the existing case
        tokens = [[token] for token in tokens]
    for j in range(mxlen):
        if j == len(tokens):
            break
        token_features = tokens[j]
        if 'word' in vocab_keys:
            word_index = vocab_keys['word']
            words_vocab = tagger.get_vocab(vocab_type='word')
            w = token_features[word_index]
            xs[0, j] = words_vocab.get(w, 0)
            xs_lc[0, j] = words_vocab.get(w.lower(), 0)
            if 'char' in vocab_keys:
                nch = min(len(w), maxw)
                for k in range(nch):
                    chars_vocab = tagger.get_vocab(vocab_type='char')
                    xs_ch[0, j, k] = chars_vocab.get(w[k], 0)
        for key in vocab_keys:
            if not key == 'word' and not key == 'char':
                feature_index = vocab_keys[key]
                feature = token_features[feature_index]
                feature_vocab = tagger.get_vocab(vocab_type=key)
                data[key] = zero_alloc((1, mxlen), dtype=np.int)
                data[key][0, j] = feature_vocab[feature]
    data.update({'x': xs, 'x_lc': xs_lc, 'xch': xs_ch, 'lengths': lengths})
    return data

