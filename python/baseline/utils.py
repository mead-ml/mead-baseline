import numpy as np
import time
import os
import importlib


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


def revlut(lut):
    return {v: k for k, v in lut.items()}


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def lowercase(x):
    return x.lower()


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
    module_name = "%s_%s" % (module_type, model_type)
    print('Loading user model %s' % module_name)
    mod = importlib.import_module(module_name)
    return mod


def create_user_classifier_model(w2v, labels, **kwargs):
    """Create a user-defined classifier model

    This creates an unstructured prediction classification model defined by the user.
    It first imports a module that must exist in the `PYTHONPATH`, with a named defined as
    `classifier_{model_type}.py`.  Once created, this user-defined model can be trained within
    the existing training programs

    :param w2v: Some type of word vectors
    :param labels: The categorical label types
    :param kwargs:
    :return: A user-defined model
    """
    model_type = kwargs['model_type']
    mod = import_user_module("classifier", model_type)
    return mod.create_model(w2v, labels, **kwargs)


def load_user_classifier_model(outname, **kwargs):
    """Loads a user-defined classifier model

    This loads a previously serialized unstructured prediction classification model defined by the user.
    It first imports a module that must exist in the `PYTHONPATH`, with a named defined as
    `classifier_{model_type}.py`.  Once loaded, this user-defined model can be used within the driver programs

    :param outname: The name of the file where the model is serialized
    :param kwargs:
    :return: A user-defined model
    """
    model_type = kwargs['model_type']
    mod = import_user_module("classifier", model_type)
    return mod.load_model(outname, **kwargs)


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


def create_user_tagger_model(labels, vocabs, **kwargs):
    """Create a user-defined tagger model

    This creates an structured prediction classification model defined by the user.
    It first imports a module that must exist in the `PYTHONPATH`, with a named defined as
    `tagger_{model_type}.py`.  Once created, this user-defined model can be trained within
    the existing training programs

    :param w2v: Some type of word vectors
    :param labels: The categorical label types
    :param kwargs:
    :return: A user-defined model
    """
    model_type = kwargs['model_type']
    mod = import_user_module("tagger", model_type)
    return mod.create_model(labels, vocabs, **kwargs)


def load_user_tagger_model(outname, **kwargs):
    """Loads a user-defined tagger model

    This loads a previously serialized structured prediction classification model defined by the user.
    It first imports a module that must exist in the `PYTHONPATH`, with a named defined as
    `tagger_{model_type}.py`.  Once loaded, this user-defined model can be used within the driver programs

    :param outname: The name of the file where the model is serialized
    :param kwargs:
    :return: A user-defined model
    """
    model_type = kwargs['model_type']
    mod = import_user_module("tagger", model_type)
    return mod.load_model(outname, **kwargs)


def create_user_lang_model(word_vec, char_vec, **kwargs):
    model_type = kwargs['model_type']
    mod = import_user_module('lang', model_type)
    return mod.create_model(word_vec, char_vec, **kwargs)


def create_user_seq2seq_model(input_embedding, output_embedding, **kwargs):
    """Create a user-defined encoder-decoder model

    This creates an encoder-decoder model defined by the user.
    It first imports a module that must exist in the `PYTHONPATH`, with a named defined as
    `tagger_{model_type}.py`.  Once created, this user-defined model can be trained within
    the existing training programs

    :param w2v: Some type of word vectors
    :param labels: The categorical label types
    :param kwargs:
    :return: A user-defined model
    """
    model_type = kwargs['model_type']
    mod = import_user_module("seq2seq", model_type)
    return mod.create_model(input_embedding, output_embedding, **kwargs)


def load_user_seq2seq_model(outname, **kwargs):
    """Loads a user-defined encoder-decoder model

    This loads a previously serialized encoder-decoder model defined by the user.
    It first imports a module that must exist in the `PYTHONPATH`, with a named defined as
    `seq2seq_{model_type}.py`.  Once loaded, this user-defined model can be used within the driver programs

    :param outname: The name of the file where the model is serialized
    :param kwargs:
    :return: A user-defined model
    """
    model_type = kwargs['model_type']
    mod = import_user_module("seq2seq", model_type)
    return mod.load_model(outname, **kwargs)


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


# Get a sparse index (dictionary) of top values
# Note: mutates input for efficiency
def topk(k, probs):

    lut = {}
    i = 0

    while i < k:
        idx = np.argmax(probs)
        lut[idx] = probs[idx]
        probs[idx] = 0
        i += 1
    return lut

#  Prune all elements in a large probability distribution below the top K
#  Renormalize the distribution with only top K, and then sample n times out of that
def beam_multinomial(k, probs):

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


# Turn a sequence of IOB chunks into single tokens
def to_spans(sequence, lut, strict_iob2=False):

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


def f_score(overlap_count, gold_count, guess_count, f=1):
    beta_sq = f*f
    if guess_count == 0: return 0.0
    precision = overlap_count / float(guess_count)
    recall = overlap_count / float(gold_count)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    f = (1. + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return f
