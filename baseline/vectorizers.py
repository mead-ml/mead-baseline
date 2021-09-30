import collections
import copy
import tempfile
import unicodedata
import re
import json
from typing import Tuple, List, Iterable, Set, Dict
import numpy as np

from functools import lru_cache
from eight_mile.downloads import open_file_or_url, get_file_or_url
from eight_mile.utils import exporter, optional_params, listify, register, Offsets, is_sequence, pads
from baseline.utils import import_user_module

try:
    import regex
except:
    # If this doesnt work, no GPT2
    pass

try:
    # If this doesnt work, no XLM-R
    import sentencepiece as spm
except:
    pass

__all__ = []
export = exporter(__all__)


@export
class Vectorizer:

    def __init__(self):
        pass

    def run(self, tokens, vocab):
        pass

    def count(self, tokens):
        pass

    def get_dims(self) -> Tuple[int]:
        pass

    def valid_label_indices(self, tokens: Iterable) -> List[int]:
        pass

    def iterable(self, tokens):
        pass

    def reset(self):
        pass


MEAD_VECTORIZERS = {}


@export
@optional_params
def register_vectorizer(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, MEAD_VECTORIZERS, name, 'vectorizer')


@export
def identity_trans_fn(x):
    return x


@export
class AbstractVectorizer(Vectorizer):

    def __init__(self, transform_fn=None, emit_begin_tok=[], emit_end_tok=[]):
        super().__init__()
        self.transform_fn = identity_trans_fn if transform_fn is None else transform_fn
        self.emit_begin_tok = listify(emit_begin_tok)
        self.emit_end_tok = listify(emit_end_tok)

    def iterable(self, tokens):
        """Produce an iterable of segmented tokens from an iterable input

        The tokens here could be subwords, the identity, or some other transform, this is really
        up to the implementation, but the intent here is that the items yielded are the underlying
        atoms, meaning that there is no processing left to do to convert them to integer values other
        than to lookup their values in a word-to-index lookup table

        :param tokens: An iterable of tokens
        :return: Generator for atoms
        """
        for t in self.emit_begin_tok:
            yield t

        for tok in tokens:
            yield self.transform_fn(tok)

        for t in self.emit_end_tok:
            yield t

    def _next_element(self, tokens, vocab):
        """This function transforms non "atomic" input to its elements and yields integer values

        Because this function requires a vocab, it cannot be used during counting (which is responsible for producing
        the text atomic words (or subwords) that may be used for vocabulary tabulation
        :param tokens: An iterable of tokens
        :param vocab:
        :return: Generator for integer values that can be directly used in Embeddings
        """
        for atom in self.iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                value = vocab.get('<UNK>', -1)
                if value == -1:
                    break
            yield value

    def valid_label_indices(self, tokens: Iterable) -> List[int]:
        """Produce the indices in an iterable containing valid labels only

        For instance, if the vectorizer deals with sub-words, this function will return
        the leader token indices
        :param tokens:
        :return:
        """
        try:
            return list(range(len(tokens)))
        except TypeError:
            return [i for i, _ in enumerate(tokens)]


@export
@register_vectorizer(name='token1d')
class Token1DVectorizer(AbstractVectorizer):

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('transform_fn'), kwargs.get('emit_begin_tok', []), kwargs.get('emit_end_tok', []))
        self.time_reverse = kwargs.get('rev', False)
        self.mxlen = kwargs.get('mxlen', -1)
        self.max_seen = 0

    def count(self, tokens):
        """Count (tabulate) the "atoms" in this tokens stream

        This method converts each token to its atoms (e.g. subwords, or transformed case tokens), and gives back
        a frequency table tabulated from the input

        :param tokens: An iterable of string tokens
        :return: A frequency table of atoms
        """
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def reset(self):
        """Reset allows the vectorizer to reset any critical information from scratch

        In this implementation, the only critical items are the max length of the temporal stream allowable and the
        maximum attested temporal stream length

        :return: None
        """
        self.mxlen = -1
        self.max_seen = -1

    def run(self, tokens, vocab):
        """Convert an iterable token stream to an integer padded up to the maximum length `mxlen)

        :param tokens: An iterable token stream
        :param vocab: A word-to-integer index
        :return: A (padded) vector and the valid (unpadded length)
        """
        if self.mxlen < 0:
            self.mxlen = self.max_seen

        vec1d = pads(self.mxlen, dtype=int)
        i = 0
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
        valid_length = i + 1

        if self.time_reverse:
            vec1d = vec1d[::-1]
            return vec1d, None
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


def _token_iterator(vectorizer, tokens):
    for tok in tokens:
        token = []
        for field in vectorizer.fields:
            if isinstance(tok, dict):
                token += [vectorizer.transform_fn(tok[field])]
            else:
                token += [vectorizer.transform_fn(tok)]
        yield vectorizer.delim.join(token)


@export
@register_vectorizer(name='dict1d')
class Dict1DVectorizer(Token1DVectorizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for t in _token_iterator(self, tokens):
            yield t
        for t in self.emit_end_tok:
            yield t
@export
@register_vectorizer(name='single-item-dict1d')
class SingleItemDict1DVectorizer(Token1DVectorizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('field', kwargs.get('fields', 'text'))

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for tok in tokens:
            yield tok[self.field]
        for t in self.emit_end_tok:
            yield t


@export
@register_vectorizer(name='int-identity-dict1d')
class IntIdentityDict1DVectorizer(Token1DVectorizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('field', kwargs.get('fields', 'text'))

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for tok in tokens:
            yield tok[self.field]
        for t in self.emit_end_tok:
            yield t

    def _next_element(self, tokens, vocab):
        """This function transforms non "atomic" input to its elements and yields integer values

        Because this function requires a vocab, it cannot be used during counting (which is responsible for producing
        the text atomic words (or subwords) that may be used for vocabulary tabulation
        :param tokens: An iterable of tokens
        :param vocab:
        :return: Generator for integer values that can be directly used in Embeddings
        """
        for value in self.iterable(tokens):
            if value == -1:
                break
            yield int(value)



@export
class AbstractCharVectorizer(AbstractVectorizer):

    #def __init__(self, transform_fn=None, emit_begin_tok=[], emit_end_tok=[]):
    #    super().__init__(transform_fn, emit_begin_tok, emit_end_tok)

    def _next_element(self, tokens, vocab):
        OOV = vocab['<UNK>']
        EOW = vocab.get('<EOW>', vocab.get(' ', Offsets.PAD))
        for token in self.iterable(tokens):
            for ch in token:
                yield vocab.get(ch, OOV)
            yield EOW

    def valid_label_indices(self, tokens: Iterable) -> List[int]:
        try:
            return list(range(len(tokens)))
        except TypeError:
            return [i for i, _ in enumerate(tokens)]


@export
@register_vectorizer(name='char2d')
class Char2DVectorizer(AbstractCharVectorizer):

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('transform_fn'), kwargs.get('emit_begin_tok', []), kwargs.get('emit_end_tok', []))
        self.mxlen = kwargs.get('mxlen', -1)
        self.mxwlen = kwargs.get('mxwlen', -1)
        self.max_seen_tok = 0
        self.max_seen_char = 0

    def count(self, tokens):
        seen_tok = 0
        counter = collections.Counter()
        for token in self.iterable(tokens):
            self.max_seen_char = max(self.max_seen_char, len(token))
            seen_tok += 1
            for ch in token:
                counter[ch] += 1
            counter['<EOW>'] += 1
        self.max_seen_tok = max(self.max_seen_tok, seen_tok)
        return counter

    def reset(self):
        self.mxlen = -1
        self.mxwlen = -1
        self.max_seen_tok = 0
        self.max_seen_char = 0

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen_tok
        if self.mxwlen < 0:
            self.mxwlen = self.max_seen_char

        EOW = vocab.get('<EOW>', vocab.get(' ', Offsets.PAD))

        vec2d = pads((self.mxlen, self.mxwlen), dtype=int)
        i = 0
        j = 0
        over = False
        for atom in self._next_element(tokens, vocab):
            if over:
                # If if we have gone over mxwlen burn tokens until we hit end of word
                if atom == EOW:
                    over = False
                continue
            if i == self.mxlen:
                break
            if atom == EOW:
                i += 1
                j = 0
                continue
            elif j == self.mxwlen:
                over = True
                i += 1
                j = 0
                continue
            else:
                vec2d[i, j] = atom
                j += 1
        valid_length = i
        return vec2d, valid_length

    def get_dims(self):
        return self.mxlen, self.mxwlen


@export
@register_vectorizer(name='dict2d')
class Dict2DVectorizer(Char2DVectorizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for t in _token_iterator(self, tokens):
            yield t
        for t in self.emit_end_tok:
            yield t

@export
@register_vectorizer(name='char1d')
class Char1DVectorizer(AbstractCharVectorizer):

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('transform_fn'))
        self.mxlen = kwargs.get('mxlen', -1)
        self.time_reverse = kwargs.get('rev', False)
        self.max_seen_tok = 0

    def count(self, tokens):
        seen_tok = 0
        counter = collections.Counter()
        for token in self.iterable(tokens):
            seen_tok += 1
            for ch in token:
                counter[ch] += 1
                seen_tok += 1
            counter['<EOW>'] += 1
            seen_tok += 1

        self.max_seen_tok = max(self.max_seen_tok, seen_tok)
        return counter

    def reset(self):
        self.mxlen = -1
        self.max_seen_tok = 0

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen_tok

        vec1d = pads(self.mxlen, dtype=int)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
        if self.time_reverse:
            vec1d = vec1d[::-1]
            return vec1d, None
        return vec1d, i + 1

    def get_dims(self):
        return self.mxlen,


@register_vectorizer(name='ngram')
class TextNGramVectorizer(Token1DVectorizer):
    def __init__(self, filtsz=3, joiner='@@', transform_fn=None, pad='<PAD>', **kwargs):
        super().__init__(**kwargs)
        self.filtsz = filtsz
        self.pad = pad
        self.joiner = joiner
        self.transform_fn = identity_trans_fn if transform_fn is None else transform_fn

    def iterable(self, tokens):
        nt = len(tokens)
        valid_range = nt - self.filtsz + 1
        for i in range(valid_range):
            chunk = tokens[i:i+self.filtsz]
            yield self.joiner.join(chunk)

    def get_padding(self):
        return [self.pad] * (self.filtsz // 2)

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        zp = self.get_padding()
        vec2d = pads(self.mxlen, dtype=int)
        padded_tokens = zp + tokens + zp
        for i, atom in enumerate(self._next_element(padded_tokens, vocab)):
            if i == self.mxlen:
                break
            vec2d[i] = atom

        lengths = min(self.mxlen, len(tokens))
        if self.time_reverse:
            vec2d = vec2d[::-1]
            return vec2d, None
        return vec2d, lengths


@register_vectorizer(name='dict-ngram')
class DictTextNGramVectorizer(TextNGramVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        if len(self.fields) > 1:
            raise Exception("Multifield N-grams arent supported right now")
        self.delim = kwargs.get('token_delim', '@@')

    def get_padding(self):
        return [{self.fields[0]: self.pad}] * (self.filtsz // 2)

    def _tx(self, tok):
        if tok == '<PAD>' or tok == '<UNK>':
            return tok
        return self.transform_fn(tok)

    def iterable(self, tokens):
        nt = len(tokens)
        if isinstance(tokens[0], collections.Mapping):
            token_list = [self._tx(tok[self.fields[0]]) for tok in tokens]
        else:
            token_list = [self._tx(tok) for tok in tokens]

        for i in range(nt - self.filtsz + 1):
            chunk = token_list[i:i+self.filtsz]
            yield self.joiner.join(chunk)


@export
class SavableFastBPE:
    """
    Use fastBPE for subwords.  If you want to use this class, make sure you have it installed
    """
    def __init__(self, codes_path, vocab_path):
        from fastBPE import fastBPE
        codes_path = get_file_or_url(codes_path)
        vocab_path = get_file_or_url(vocab_path)
        with open(codes_path, 'rb') as rf:
            self.codes = rf.read()
        with open(vocab_path, 'rb') as rf:
            self.vocab = rf.read()
        self.bpe = fastBPE(codes_path, vocab_path)

    @property
    def subword_sentinel(self):
        return "@@"

    def __getstate__(self):
        return {'codes': self.codes, 'vocab': self.vocab}

    def __setstate__(self, state):
        from fastBPE import fastBPE
        with tempfile.NamedTemporaryFile() as codes, tempfile.NamedTemporaryFile() as vocab:
            codes.write(state['codes'])
            vocab.write(state['vocab'])
            self.bpe = fastBPE(codes.name, vocab.name)

    def apply(self, sentences):
        return self.bpe.apply(sentences)


@export
class HasPredefinedVocab:
    """Define an interface for predefined vocabs.  Using a sub-class of this means readers dont need to collect a vocab
    """

    def read_vocab(self, file_or_url) -> Dict[str, int]:
        """Read a pre-defined vocab from a file and give back a vocab of (sub)words to integer values

        If the file is presented as a URL, it will be downloaded first

        :param file_or_url: A file or URL
        :return: A vocabular of word to indices
        """
    @property
    def vocab(self):
        pass

    @property
    def special_tokens(self) -> Set[str]:
        """Return a set of special tokens"""


@export
class HasSubwordTokens(HasPredefinedVocab):

    @property
    def subword_sentinel(self):
        """Indicates the special token used to demarcate subwords

        :return:
        """
    def valid_label_indices(self, tokens: Iterable) -> List[int]:
        """Give back the indices that would contain "valid" labels (i.e. leader tokens)

        When a word is split into sub-words, usually, the first token is used as the indicator for
        a problem where there is one label per word usually.  This function knows how to get back those
        indices

        :param tokens:
        :return:
        """

# TODO: Most of our classes have the `Vectorizer` part of the name at the end
@export
@register_vectorizer(name='bpe1d')
class BPEVectorizer1D(AbstractVectorizer, HasSubwordTokens):
    """Define a Baseline Vectorizer for BPE using fastBPE (https://github.com/glample/fastBPE) or
    subword-nmt (https://github.com/rsennrich/subword-nmt)

    If you use tokens=bpe, this vectorizer is used, and so then there is a
    dependency on either fastBPE or subword-NMT.  To configure which, set boolean `use_fast_bpe`
    (defaults to True).  If using subword-NMT, you can make use of out-of-vocabulary glossaries.

    The implementation here subclasses the subword NMT `BPE` class to allow it to read in
    `fastBPE` files.  The difference is those files have a 3rd column, and they contain do not
    specify a head of #version 0.2 even though they are v0.2 files.
    See https://github.com/rsennrich/subword-nmt/issues/76

    To use BPE, we assume that a Dictionary of codes and vocab was already created

    """
    def __init__(self, **kwargs):
        """Loads a BPE tokenizer"""
        super().__init__(kwargs.get('transform_fn'), kwargs.get('emit_begin_tok', []), kwargs.get('emit_end_tok', []))
        self.max_seen = 128
        self.model_file = kwargs.get('model_file')
        self.vocab_file = kwargs.get('vocab_file')
        use_fast_bpe = kwargs.get('use_fast_bpe', True)
        self._special_tokens = {"[CLS]", "<unk>", "<EOS>", "<UNK>", "<s>", "</s>"}
        self._extra_tokens = kwargs.get('extra_tokens', ['[CLS]', '[MASK]'])
        self.tokenizer = SavableFastBPE(self.model_file, self.vocab_file)
        self.mxlen = kwargs.get('mxlen', -1)
        vocab_list = self.read_vocab(self.vocab_file)
        self._vocab = {k: i for i, k in enumerate(vocab_list)}

    @property
    def vocab(self):
        return self._vocab

    @property
    def special_tokens(self) -> Set[str]:
        return self._special_tokens

    @property
    def extra_tokens(self) -> List[str]:
        return self._extra_tokens

    @property
    def subword_sentinel(self):
        return getattr(self.tokenizer, "subword_sentinel", "@@")

    def valid_label_indices(self, tokens: Iterable) -> List[int]:
        indices = []
        in_subword = False
        for i, token in enumerate(tokens):
            if token in self.special_tokens:
                in_subword = False
                continue
            if not in_subword:
                indices.append(i)
                if token.endswith(self.subword_sentinel):
                    in_subword = True
            else:
                if not token.endswith(self.subword_sentinel):
                    in_subword = False
        return indices

    def read_vocab(self, file_or_url):
        vocab = [] + Offsets.VALUES + self._extra_tokens
        with open_file_or_url(file_or_url, "r") as f:
            for line in f.readlines():
                token = line.split()[0].strip()
                vocab.append(token)
        return vocab

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def iterable(self, tokens):
        extra = self._extra_tokens if hasattr(self, '_extra_tokens') else ['[CLS]', '[MASK]']
        outside_vocab = set(Offsets.VALUES + extra)
        for t in self.emit_begin_tok:
            yield t

        for t in tokens:
            if t in outside_vocab:
                yield t
            elif t == '<unk>':
                yield Offsets.VALUES[Offsets.UNK]
            elif t == '<eos>':
                yield Offsets.VALUES[Offsets.EOS]
            else:
                subwords = self.tokenizer.apply([self.transform_fn(t)])[0].split()
                for x in subwords:
                    yield x
        for t in self.emit_end_tok:
                yield t

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom, Offsets.UNK)  # This shouldnt actually happen
            yield value

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        i = 0
        vec1d = pads(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= len(self.emit_end_tok)
                for j, x in enumerate(self.emit_end_tok):
                    vec1d[i + j] = vocab.get(x)
                i = self.mxlen - 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


@export
@register_vectorizer(name='bpe-secondary-feature-dict1d')
class BPESecondaryFeatureDict1DVectorizer(BPEVectorizer1D):
    """We need to split on the primary feature but use a secondary feature's value

    Some options concern what to do with the non primary index.  For a label, this would typically
    be a `<PAD>` token in the non first position of a sub-word, but that may not be desirable here

    To support bot ways, there is an optional `apply_all_subwords`, which defaults to True.  If this
    is turned on, it means that we want to use the feature value of


    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field'))
        self.primary_feature = kwargs.get('primary_feature', 'text')
        self.apply_all_subwords = kwargs.get('apply_all_subwords', True)

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for t in tokens:
            t_word = t[self.primary_feature]
            t_feature = t[self.field]
            if t_word in Offsets.VALUES:
                yield t_feature
            elif t == '<unk>':
                yield t_feature
            elif t == '<eos>':
                yield t_feature
            else:
                subwords = self.tokenizer.apply([t_word])[0].split()
                if self.apply_all_subwords:
                    subwords = [t_feature] * len(subwords)
                else:
                    subwords = [Offsets.VALUES[Offsets.PAD]] * len(subwords)
                    subwords[0] = t_feature
                for x in subwords:
                    yield x
        for t in self.emit_end_tok:
            yield t

    def run(self, tokens, vocab):
        return super().run(tokens, vocab)



@export
@register_vectorizer(name='bpe-label-dict1d')
class BPELabelDict1DVectorizer(BPEVectorizer1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field', 'text'))
        self.label = kwargs.get('label', 'label')
        #self.emit_begin_tok = kwargs.get('emit_begin_tok')
        #self.emit_end_tok = kwargs.get('emit_end_tok')

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for t in tokens:
            t_word = t[self.field]
            t_label = t[self.label]
            if t_word in Offsets.VALUES:
                yield t_label
            elif t == '<unk>':
                yield t_label
            elif t == '<eos>':
                yield t_label
            else:
                subwords = self.tokenizer.apply([self.transform_fn(t_word)])[0].split()
                subwords = [Offsets.VALUES[Offsets.PAD]] * len(subwords)
                subwords[0] = t_label
                for x in subwords:
                    yield x
        for t in self.emit_end_tok:
            yield t

    def run(self, tokens, vocab):
        return super().run(tokens, vocab)


@export
@register_vectorizer(name='bpe-dict1d')
class BPEDict1DVectorizer(BPEVectorizer1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field', 'text'))
        self.delim = kwargs.get('token_delim', '~~')

    def iterable(self, tokens):
        tok = [t[self.field] if isinstance(t, dict) else t for t in tokens]

        return super().iterable(tok)


# This code is borrowed from the official BERT rep and slightly modified
# https://github.com/google-research/bert/
def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


BERT_VOCAB = None


@export
def load_bert_vocab(vocab_file):
    global BERT_VOCAB
    if BERT_VOCAB is not None:
        return BERT_VOCAB

    vocab = collections.OrderedDict()
    index = 0
    with open_file_or_url(vocab_file, "r") as rf:
        for line in rf:
            token = convert_to_unicode(line)
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    BERT_VOCAB = vocab
    return vocab


class FullTokenizer:
    """Runs end-to-end tokenization."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_bert_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer:
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer:
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    @property
    def subword_sentinel(self):
        return "##"

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


# TODO: Most of our classes have the `Vectorizer` part of the name at the end
@register_vectorizer(name='wordpiece1d')
class WordpieceVectorizer1D(AbstractVectorizer, HasSubwordTokens):

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('transform_fn'), kwargs.get('emit_begin_tok', ['[CLS]']), kwargs.get('emit_end_tok', ['[SEP]']))
        self.max_seen = 128
        self.tokenizer = WordpieceTokenizer(self.read_vocab(kwargs.get('vocab_file')))
        self.mxlen = kwargs.get('mxlen', -1)
        self.dtype = kwargs.get('dtype', 'int')
        self._special_tokens = {"[CLS]", "<unk>", "<EOS>"}

    def read_vocab(self, file):
        return load_bert_vocab(file)

    @property
    def subword_sentinel(self):
        return getattr(self.tokenizer, "subword_sentinel", "##")

    @property
    def special_tokens(self) -> Set[str]:
        return self._special_tokens

    def valid_label_indices(self, tokens: Iterable) -> List[int]:
        return [i for i, t in enumerate(tokens) if not t.startswith(self.subword_sentinel) and t not in self.special_tokens]

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for tok in tokens:
            if tok == '<unk>':
                yield '[UNK]'
            elif tok == '<EOS>':
                yield '[SEP]'
            else:
                for subtok in self.tokenizer.tokenize(self.transform_fn(tok)):
                    yield subtok
        for t in self.emit_end_tok:
            yield t

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                value = vocab['[UNK]']
            yield value

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = pads(self.mxlen, dtype=self.dtype)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= len(self.emit_end_tok)
                for j, x in enumerate(self.emit_end_tok):
                    vec1d[i + j] = vocab.get(x, vocab['[UNK]'])
                i = self.mxlen - 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    @property
    def vocab(self):
        return self.tokenizer.vocab

    def get_dims(self):
        return self.mxlen,



@export
@register_vectorizer(name='wordpiece-secondary-feature-dict1d')
class WordpieceSecondaryFeatureDict1DVectorizer(WordpieceVectorizer1D):
    """We need to split on the primary feature but use a secondary feature's value

    Some options concern what to do with the non primary index.  For a label, this would typically
    be a `<PAD>` token in the non first position of a sub-word, but that may not be desirable here

    To support bot ways, there is an optional `apply_all_subwords`, which defaults to True.  If this
    is turned on, it means that we want to use the feature value of


    """
    def __init__(self, **kwargs):
        kwargs['emit_begin_tok'] = kwargs.get('emit_begin_tok', [Offsets.VALUES[Offsets.PAD]])
        kwargs['emit_end_tok'] = kwargs.get('emit_end_tok', [Offsets.VALUES[Offsets.PAD]])
        super().__init__(**kwargs)
        self.pad_value = kwargs.get('pad_value', Offsets.VALUES[Offsets.PAD])

        self.field = kwargs.get('fields', kwargs.get('field'))
        self.primary_feature = kwargs.get('primary_feature', 'text')
        self.apply_all_subwords = kwargs.get('apply_all_subwords', True)

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for t in tokens:
            t_word = t[self.primary_feature]
            t_feature = t[self.field]
            if t_word in Offsets.VALUES:
                yield t_feature
            elif t == '<unk>':
                yield t_feature
            elif t == '<eos>':
                yield t_feature
            else:
                subwords = self.tokenizer.tokenize(t_word)
                if self.apply_all_subwords:
                    subwords = [t_feature] * len(subwords)
                else:
                    subwords = [self.pad_value] * len(subwords)
                    subwords[0] = t_feature
                for x in subwords:
                    yield x
        for t in self.emit_end_tok:
            yield t

    def run(self, tokens, vocab):
        return super().run(tokens, vocab)


@register_vectorizer(name='wordpiece-label-dict1d')
class WordpieceLabelDict1DVectorizer(WordpieceVectorizer1D):

    def __init__(self, **kwargs):
        kwargs['emit_begin_tok'] = kwargs.get('emit_begin_tok', [Offsets.VALUES[Offsets.PAD]])
        kwargs['emit_end_tok'] = kwargs.get('emit_end_tok', [Offsets.VALUES[Offsets.PAD]])
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field', 'text'))
        self.label = kwargs.get('label', 'label')

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for t in tokens:
            t_word = t[self.field]
            t_label = t[self.label]
            subwords = [x for x in self.tokenizer.tokenize(t_word)]
            subwords = [Offsets.VALUES[Offsets.PAD]] * len(subwords)
            # TODO: The tokenizer sometimes cuts up the token and leaves nothing
            # how to handle this since we cannot get anything for it
            if len(subwords):
                subwords[0] = t_label
            for x in subwords:
                yield x
        for t in self.emit_end_tok:
            yield t

    def run(self, tokens, vocab):
        return super().run(tokens, vocab)



@export
@register_vectorizer(name='wordpiece-int-identity-dict1d')
class WordpieceIntIdentityDict1DVectorizer(WordpieceSecondaryFeatureDict1DVectorizer):
    """We need to split on the primary feature but use a secondary feature's value

    Some options concern what to do with the non primary index.  For a label, this would typically
    be a `<PAD>` token in the non first position of a sub-word, but that may not be desirable here

    To support bot ways, there is an optional `apply_all_subwords`, which defaults to True.  If this
    is turned on, it means that we want to use the feature value of


    """
    def __init__(self, **kwargs):
        kwargs['emit_begin_tok'] = kwargs.get('emit_begin_tok', [0])
        kwargs['emit_end_tok'] = kwargs.get('emit_end_tok', [0])
        kwargs['pad_value'] = kwargs.get('pad_value', 0)
        super().__init__(**kwargs)

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            yield int(atom)



@register_vectorizer(name='wordpiece-dict1d')
class WordpieceDict1DVectorizer(WordpieceVectorizer1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field', 'text'))
        self.delim = kwargs.get('token_delim', '~~')

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for t in tokens:
            tok = t[self.field] if isinstance(t, dict) else t
            if tok == '<unk>':
                yield '[UNK]'
            elif tok == '<EOS>':
                yield '[SEP]'
            else:
                for subtok in self.tokenizer.tokenize(self.transform_fn(tok)):
                    yield subtok
        for t in self.emit_end_tok:
            yield t



@register_vectorizer(name="pass-through")
class PassThroughVectorizer(AbstractVectorizer):
    def __init__(self, feature_size = None, **kwargs):
        self.feature_size = feature_size
        self.max_seen = 0
        self.mxlen = -1

    def count(self, tokens):
        return {}
    def reset(self):
        """Reset allows the vectorizer to reset any critical information from scratch
        In this implementation, the only critical items are the max length of the temporal stream allowable and the
        maximum attested temporal stream length
        :return: None
        """
        self.mxlen = -1
        self.max_seen = -1

    def run(self, tokens, _):
        self.feature_size = len(tokens)
        return tokens, self.feature_size


@register_vectorizer(name="pass-through-dict")
class PassThroughDict1DVectorizer(PassThroughVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def count(self, tokens):
        self.max_seen = max(self.max_seen, len(tokens))

    def run(self, tokens, _):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        features = np.stack([[t[self.fields[i]] for i in range(len(self.fields))] for t in tokens])
        self.feature_size = features.shape[-1]
        vec = np.zeros((self.mxlen, self.feature_size), dtype=np.float32)
        vec[:len(features)] = features
        return vec, len(features)

# The code below is modified from:
# https://raw.githubusercontent.com/openai/gpt-2/master/src/encoder.py"""

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in regex.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def encode_subword(self, text):
        bpe_tokens = []
        for token in regex.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


@register_vectorizer(name='gpt2-bpe1d')
class GPT2Vectorizer1D(AbstractVectorizer, HasSubwordTokens):

    def __init__(self, **kwargs):
        """Loads a BPE tokenizer"""
        super().__init__(kwargs.get('transform_fn'), kwargs.get('emit_begin_tok', []), kwargs.get('emit_end_tok', []))
        self.max_seen = 128
        self.model_file = kwargs.get('model_file')
        self.vocab_file = kwargs.get('vocab_file')
        self._special_tokens = {"<unk>", "<pad>", "<s>", "</s>"}

        with open_file_or_url(self.vocab_file, "r") as f:
            vocab = json.load(f)

        fpath = get_file_or_url(self.model_file)
        with open(fpath, "r", encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
        self.tokenizer = Encoder(
            encoder=vocab,
            bpe_merges=bpe_merges,
        )

        self.mxlen = kwargs.get('mxlen', -1)
        Offsets.INDICES['PAD'] = self.vocab['<pad>']
        Offsets.INDICES['GO'] = self.vocab['<s>']
        Offsets.INDICES['EOS'] = self.vocab['</s>']
        Offsets.INDICES['UNK'] = self.vocab['<unk>']
        #Offsets.VALUES['PAD'] = '<pad>'
        #Offsets.VALUES['GO'] = '<s>'
        #Offsets.VALUES['EOS'] = '</s>'
        #Offsets.VALUES['UNK'] = ['<unk>']

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def special_tokens(self) -> Set[str]:
        return self._special_tokens

    @property
    def subword_sentinel(self):
        return getattr(self.tokenizer, "subword_sentinel", "@@")

    def valid_label_indices(self, tokens: Iterable) -> List[int]:
        indices = []
        in_subword = False
        for i, token in enumerate(tokens):
            if token in self.special_tokens:
                in_subword = False
                continue
            if not in_subword:
                indices.append(i)
                if token.endswith(self.subword_sentinel):
                    in_subword = True
            else:
                if not token.endswith(self.subword_sentinel):
                    in_subword = False
        return indices

    def read_vocab(self, file_or_url):
        with open_file_or_url(file_or_url, "r") as f:
            self._vocab = json.load(f)

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t

        if not isinstance(tokens, str):
            tokens = ' '.join(tokens)

        bpe_tokens = self.tokenizer.encode_subword(tokens)
        for t in bpe_tokens:
            yield t
        for t in self.emit_end_tok:
            yield t

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom, vocab.get(Offsets.VALUES[Offsets.UNK]))
            yield value

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = np.ones(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= len(self.emit_end_tok)
                for j, x in enumerate(self.emit_end_tok):
                    vec1d[i + j] = vocab.get(x)
                i = self.mxlen - 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


@register_vectorizer(name='xlmr-spm1d')
class XLMRSentencePieceVectorizer1D(AbstractVectorizer, HasSubwordTokens):
    SPECIAL_TOKENS = {"<unk>", "<pad>", "<s>", "</s>"}
    VOCAB_OFFSETS = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3}

    def _REWIRE_GLOBAL_OFFSETS(self):

        Offsets.INDICES['PAD'] = self._vocab['<pad>']
        Offsets.INDICES['GO'] = self._vocab['<s>']
        Offsets.INDICES['EOS'] = self._vocab['</s>']
        Offsets.INDICES['UNK'] = self._vocab['<unk>']
        Offsets.VALUES = ["<GO>", "<PAD>", "<EOS>", "<UNK>"]


    def __init__(self,
                 **kwargs):
        """Loads a BPE tokenizer"""
        super().__init__(kwargs.get('transform_fn'),
                         kwargs.get('emit_begin_tok', []),
                         kwargs.get('emit_end_tok', []))
        self.max_seen = 512
        self.model_file = kwargs.get('model_file')
        self.tokenizer = spm.SentencePieceProcessor(self.model_file)


        self._special_tokens = XLMRSentencePieceVectorizer1D.SPECIAL_TOKENS
        self._vocab = XLMRSentencePieceVectorizer1D.VOCAB_OFFSETS
        self.mxlen = kwargs.get('mxlen', -1)
        self._REWIRE_GLOBAL_OFFSETS()

        for i in range(3, len(self.tokenizer)):
            v = self.tokenizer.IdToPiece(i)
            self._vocab[v] = i + 1
        self._vocab['<mask>'] = len(self.tokenizer) + 1

    def __setstate__(self, d):
        self.__dict__ = d
        self._REWIRE_GLOBAL_OFFSETS()

    @property
    def vocab(self):
        return self._vocab

    @property
    def special_tokens(self) -> Set[str]:
        return self._special_tokens

    @property
    def subword_sentinel(self):
        return getattr(self.tokenizer, "subword_sentinel", "@@")

    def valid_label_indices(self, tokens: Iterable) -> List[int]:
        indices = []
        in_subword = False
        for i, token in enumerate(tokens):
            if token in self.special_tokens:
                in_subword = False
                continue
            if not in_subword:
                indices.append(i)
                if token.endswith(self.subword_sentinel):
                    in_subword = True
            else:
                if not token.endswith(self.subword_sentinel):
                    in_subword = False
        return indices

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t

        if not isinstance(tokens, str):
            tokens = ' '.join(tokens)

        spm_tokens = self.tokenizer.EncodeAsPieces(tokens)
        for t in spm_tokens:
            yield t
        for t in self.emit_end_tok:
            yield t

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom, Offsets.UNK)
            yield value

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = pads(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= len(self.emit_end_tok)
                for j, x in enumerate(self.emit_end_tok):
                    vec1d[i + j] = vocab.get(x)
                i = self.mxlen - 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


@export
@register_vectorizer(name='xlmr-spm-dict1d')
class XLMRSentencePieceDict1DVectorizer(XLMRSentencePieceVectorizer1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field', 'text'))
        self.delim = kwargs.get('token_delim', '~~')

    def iterable(self, tokens):
        tok = [t[self.field] if isinstance(t, dict) else t for t in tokens]

        return super().iterable(tok)

@export
@register_vectorizer(name='xlmr-spm-label-dict1d')
class XLMRSentencePieceLabelDict1DVectorizer(XLMRSentencePieceVectorizer1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field', 'text'))
        self.label = kwargs.get('label', 'label')

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t
        for t in tokens:
            t_word = t[self.field]
            t_label = t[self.label]
            if t_word in Offsets.VALUES:
                yield t_label
            elif t == '<unk>':
                yield t_label
            elif t == '<eos>':
                yield t_label
            else:
                subwords = self.tokenizer.EncodeAsPieces(t_word)  # TODO: is this right?
                #subwords = self.tokenizer.apply([self.transform_fn(t_word)])[0].split()
                subwords = [Offsets.VALUES[Offsets.PAD]] * len(subwords)
                subwords[0] = t_label
                for x in subwords:
                    yield x
        for t in self.emit_end_tok:
            yield t

    def run(self, tokens, vocab):
        return super().run(tokens, vocab)


@register_vectorizer(name='camembert-spm1d')
class CamembertSentencePieceVectorizer1D(AbstractVectorizer, HasSubwordTokens):

    SPECIAL_TOKENS = {"<unk>", "<pad>", "<s>", "</s>"}
    VOCAB_OFFSETS = {'<s>': 5, '<pad>': 1, '</s>': 6, '<unk>': 3}

    def _REWIRE_GLOBAL_OFFSETS(self):
        """This function mutates the global state for Offsets, it is required for roberta-like models

        :return:
        """
        Offsets.INDICES['PAD'] = self._vocab['<pad>']
        Offsets.INDICES['GO'] = self._vocab['<s>']
        Offsets.INDICES['EOS'] = self._vocab['</s>']
        Offsets.INDICES['UNK'] = self._vocab['<unk>']

    def __init__(self,
                 **kwargs):
        """Loads a BPE tokenizer"""
        super().__init__(kwargs.get('transform_fn'),
                         kwargs.get('emit_begin_tok', []),
                         kwargs.get('emit_end_tok', []))
        self.max_seen = 512
        self.model_file = kwargs.get('model_file')
        self.tokenizer = spm.SentencePieceProcessor(self.model_file)


        self._special_tokens = CamembertSentencePieceVectorizer1D.SPECIAL_TOKENS
        self._vocab = copy.deepcopy(CamembertSentencePieceVectorizer1D.VOCAB_OFFSETS)
        self.mxlen = kwargs.get('mxlen', -1)
        self._REWIRE_GLOBAL_OFFSETS()

        offset = len(CamembertSentencePieceVectorizer1D.VOCAB_OFFSETS)
        for i in range(len(self.tokenizer)):
            v = self.tokenizer.IdToPiece(i)
            self._vocab[v] = i + offset
        self._vocab['<mask>'] = offset + len(self.tokenizer) + 1

    def __setstate__(self, d):
        self.__dict__ = d
        self._REWIRE_GLOBAL_OFFSETS()

    @property
    def vocab(self):
        return self._vocab

    @property
    def special_tokens(self) -> Set[str]:
        return self._special_tokens

    @property
    def subword_sentinel(self):
        return getattr(self.tokenizer, "subword_sentinel", "@@")

    def valid_label_indices(self, tokens: Iterable) -> List[int]:
        indices = []
        in_subword = False
        for i, token in enumerate(tokens):
            if token in self.special_tokens:
                in_subword = False
                continue
            if not in_subword:
                indices.append(i)
                if token.endswith(self.subword_sentinel):
                    in_subword = True
            else:
                if not token.endswith(self.subword_sentinel):
                    in_subword = False
        return indices

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def iterable(self, tokens):
        for t in self.emit_begin_tok:
            yield t

        if not isinstance(tokens, str):
            tokens = ' '.join(tokens)

        spm_tokens = self.tokenizer.EncodeAsPieces(tokens)
        for t in spm_tokens:
            yield t
        for t in self.emit_end_tok:
            yield t

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom, Offsets.UNK)
            yield value

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = pads(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= len(self.emit_end_tok)
                for j, x in enumerate(self.emit_end_tok):
                    vec1d[i + j] = vocab.get(x)
                i = self.mxlen - 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


@register_vectorizer(name='bpe-nbest1d')
class BPENBestVectorizer1D(BPEVectorizer1D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nbest = kwargs.get('nbest', 3)
        self.eou_tok = kwargs.get('eou_tok', '<EOU>')

    def count(self, utts):
        from collections import Counter
        utts = utts[:self.nbest]
        seen = 0
        counter = Counter()
        tokens = []
        for utt in utts:
            tokens += utt.split() + [self.eou_tok]
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def run(self, utts, vocab):
        utts = utts[:self.nbest]
        tokens = []
        for utt in utts:
            tokens += utt.split() + [self.eou_tok]
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        i = 0
        vec1d = pads(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= len(self.emit_end_tok)
                for j, x in enumerate(self.emit_end_tok):
                    vec1d[i + j] = vocab.get(x)
                i = self.mxlen - 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length


@register_vectorizer(name='bpe-nbest2d')
class BPENBestVectorizer2D(BPENBestVectorizer1D):
    def run(self, utts, vocab):
        utts = utts[:self.nbest]
        vec2d = pads((self.nbest, self.mxlen), dtype=int)
        # ensure empty utt also has the CLS token for cls pooling
        vec2d[:, 0] = vocab.get('[CLS]', vocab.get('<s>', vocab.get('<UNK>')))
        # we use utt length=1 to represent an empty utt. it's necessary for rnn, also consistent with CLS pooling
        lengths = np.ones(self.nbest, dtype=int)
        for i, utt in enumerate(utts):
            for j, atom in enumerate(self._next_element(utt.split(), vocab)):
                if j == self.mxlen:
                    j -= len(self.emit_end_tok)
                    for k, x in enumerate(self.emit_end_tok):
                        vec2d[i, j + k] = vocab.get(x)
                    j = self.mxlen - 1
                    break
                vec2d[i, j] = atom
            lengths[i] = j + 1
        return vec2d, lengths

    def get_dims(self):
        return self.nbest, self.mxlen

@export
def create_vectorizer(**kwargs):
    vec_type = kwargs.get('vectorizer_type', kwargs.get('type', 'token1d'))
    # Dynamically load a module if its needed

    for module in listify(kwargs.get('module', kwargs.get('modules', []))):
        import_user_module(module, kwargs.get('data_download_cache'))
    Constructor = MEAD_VECTORIZERS.get(vec_type)
    return Constructor(**kwargs)

