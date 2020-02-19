import numpy as np
from baseline.utils import exporter, optional_params, listify, register, Offsets, import_user_module
import collections


__all__ = []
export = exporter(__all__)


@export
class Vectorizer(object):

    def __init__(self):
        pass

    def run(self, tokens, vocab):
        pass

    def count(self, tokens):
        pass

    def get_dims(self):
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

    def __init__(self, transform_fn=None):
        super().__init__()
        self.transform_fn = identity_trans_fn if transform_fn is None else transform_fn

    def iterable(self, tokens):
        for tok in tokens:
            yield self.transform_fn(tok)

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                value = vocab.get('<UNK>', -1)
                if value == -1:
                    break
            yield value


@export
@register_vectorizer(name='token1d')
class Token1DVectorizer(AbstractVectorizer):

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('transform_fn'))
        self.time_reverse = kwargs.get('rev', False)
        self.mxlen = kwargs.get('mxlen', -1)
        self.max_seen = 0

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def reset(self):
        self.mxlen = -1
        self.max_seen = -1

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen

        vec1d = np.zeros(self.mxlen, dtype=int)
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


@export
class GOVectorizer(Vectorizer):

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def iterable(self, tokens):
        raise Exception("Not implemented")

    def count(self, tokens):
        counter = self.vectorizer.count(tokens)
        counter['<GO>'] += 1
        return counter

    def run(self, tokens, vocab):
        vec1d, valid_length = self.vectorizer.run(tokens, vocab)
        vec1d = np.concatenate([[Offsets.GO], vec1d, [Offsets.PAD]])
        vec1d[valid_length+1] = Offsets.EOS
        return vec1d, valid_length + 2

    def get_dims(self):
        return self.vectorizer.get_dims()[0] + 2,


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
        return _token_iterator(self, tokens)



@export
class AbstractCharVectorizer(AbstractVectorizer):

    def __init__(self, transform_fn=None):
        super().__init__(transform_fn)

    def _next_element(self, tokens, vocab):
        OOV = vocab['<UNK>']
        EOW = vocab.get('<EOW>', vocab.get(' ', Offsets.PAD))
        for token in self.iterable(tokens):
            for ch in token:
                yield vocab.get(ch, OOV)
            yield EOW


@export
@register_vectorizer(name='char2d')
class Char2DVectorizer(AbstractCharVectorizer):

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('transform_fn'))
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

        vec2d = np.zeros((self.mxlen, self.mxwlen), dtype=int)
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
        return _token_iterator(self, tokens)


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

        vec1d = np.zeros(self.mxlen, dtype=int)
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
        vec2d = np.zeros(self.mxlen, dtype=int)
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



class SavableFastBPE:
    def __init__(self, codes_path, vocab_path):
        from fastBPE import fastBPE
        self.codes = open(codes_path, 'rb').read()
        self.vocab = open(vocab_path, 'rb').read()
        self.bpe = fastBPE(codes_path, vocab_path)

    def __getstate__(self):
        return {'codes': self.codes, 'vocab': self.vocab}

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile() as codes, tempfile.NamedTemporaryFile() as vocab:
            codes.write(state['codes'])
            vocab.write(state['vocab'])
            self.bpe = fastBPE(codes.name, vocab.name)

    def apply(self, sentences):
        return self.bpe.apply(sentences)


@export
@register_vectorizer(name='bpe1d')
class BPEVectorizer1D(AbstractVectorizer):
    """Define a Baseline Vectorizer for BPE using fastBPE (https://github.com/glample/fastBPE)

    If you use tokens=bpe, this vectorizer is used, and so then there is a
    dependency on fastBPE

    To use BPE, we assume that a Dictionary of codes and vocab was already created

    """
    def __init__(self, **kwargs):
        """Loads a BPE tokenizer"""
        super().__init__(kwargs.get('transform_fn'))
        self.max_seen = 128
        self.model_file = kwargs.get('model_file')
        self.vocab_file = kwargs.get('vocab_file')
        self.tokenizer = SavableFastBPE(self.model_file, self.vocab_file)
        self.mxlen = kwargs.get('mxlen', -1)
        self.vocab = {k: i for i, k in enumerate(self.read_vocab(self.vocab_file))}

    def read_vocab(self, s):
        vocab = [] + Offsets.VALUES + ['[CLS]', '[MASK]']
        with open(s, "r") as f:
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
        for t in tokens:
            if t in Offsets.VALUES:
                yield t
            elif t == '<unk>':
                yield Offsets.VALUES[Offsets.UNK]
            elif t == '<eos>':
                yield Offsets.VALUES[Offsets.EOS]
            else:
                subwords = self.tokenizer.apply([t])[0].split()
                for x in subwords:
                    yield x

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom, vocab.get(Offsets.VALUES[Offsets.UNK]))  # This shouldnt actually happen
            yield value

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = np.zeros(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


@export
@register_vectorizer(name='bpe-label-dict1d')
class BPELabelDict1DVectorizer(BPEVectorizer1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field', 'text'))
        self.label = kwargs.get('label', 'label')

    def iterable(self, tokens):
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
                subwords = self.tokenizer.apply([t_word])[0].split()
                subwords = [Offsets.VALUES[Offsets.PAD]] * len(subwords)
                subwords[0] = t_label
                for x in subwords:
                    yield x

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
        return super().iterable([t[self.field] for t in tokens])


@export
def create_vectorizer(**kwargs):
    vec_type = kwargs.get('vectorizer_type', kwargs.get('type', 'token1d'))
    # Dynamically load a module if its needed

    for module in listify(kwargs.get('module', kwargs.get('modules', []))):
        import_user_module(module, kwargs.get('data_download_cache'))
    Constructor = MEAD_VECTORIZERS.get(vec_type)
    return Constructor(**kwargs)


