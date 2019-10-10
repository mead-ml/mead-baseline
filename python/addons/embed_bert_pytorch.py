from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import unicodedata
import six
import numpy as np
from baseline.model import register_model
from baseline.reader import register_reader, CONLLSeqReader
from baseline.utils import write_json, listify
from eight_mile.embeddings import register_embeddings
from eight_mile.pytorch.embeddings import PyTorchEmbeddings
from baseline.vectorizers import register_vectorizer, AbstractVectorizer, _token_iterator
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel
from baseline.pytorch.torchy import *
from baseline.pytorch.tagger.model import TaggerModelBase
from eight_mile.pytorch.layers import tensor_and_lengths
import copy
import json
import math
import re


BERT_TOKENIZER = None


@register_vectorizer(name='wordpiece1d')
class WordPieceVectorizer1D(AbstractVectorizer):

    def __init__(self, **kwargs):
        super(WordPieceVectorizer1D, self).__init__(kwargs.get('transform_fn'))
        global BERT_TOKENIZER
        self.max_seen = 128
        handle = kwargs.get('embed_file')
        if BERT_TOKENIZER is None:
            BERT_TOKENIZER = BertTokenizer.from_pretrained(handle)
        self.tokenizer = BERT_TOKENIZER
        self.mxlen = kwargs.get('mxlen', -1)

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def iterable(self, tokens):
        yield '[CLS]'
        for tok in tokens:
            if tok == '<unk>':
                yield '[UNK]'
            elif tok == '<EOS>':
                yield '[SEP]'
            else:
                for subtok in self.tokenizer.tokenize(tok):
                    yield subtok
        yield '[SEP]'

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                value = vocab['[UNK]']
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

    def reset(self):
        self.mxlen = -1
        self.max_seen = 0


@register_vectorizer(name="dict-wordpiece1d")
class DictWordPieceVectoizer1D(WordPieceVectorizer1D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '~~')

    def iterable(self, tokens):
        return super().iterable(_token_iterator(self, tokens))


def mask_to_index(mask):
    """Convert a 1d mask to a 1d list of indices of the 1's. the result is the same shape as the mask."""
    idx = mask.nonzero()[0]
    return np.concatenate((idx, np.zeros((len(mask) - len(idx),), dtype=mask.dtype)))


@register_vectorizer(name='wordpiece1d-with-heads')
class WordPieceVectorizer1DWithHeads(WordPieceVectorizer1D):
    """This vectorizer produces the tokens and the index of the heads at the same time.

    This outputs a matrix of size [2, T] where [0, T] is the tokens and [1, T] is the mask
    """

    def iterable(self, tokens):
        yield '[CLS]', 0
        for tok in tokens:
            if tok == '<unk>':
                yield '[UNK]', 1
            elif tok == '<EOS>':
                yield '[SEP]', 0
            else:
                for i, subtok in enumerate(self.tokenizer.tokenize(tok)):
                    if i == 0:
                        yield subtok, 1
                    else:
                        yield subtok, 0
        yield '[SEP]', 0

    def _next_element(self, tokens, vocab):
        for atom, head in self.iterable(tokens):
            value = vocab.get(atom, vocab['[UNK]'])
            yield value, head

    def run(self, tokens, vocab):
        self.mxlen = self.max_seen if self.mxlen < 0 else self.mxlen
        vec1d = np.zeros(self.mxlen, dtype=np.long)
        heads = np.zeros(self.mxlen, dtype=np.long)
        for i, (atom, head) in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
            heads[i] = head
        heads = mask_to_index(heads)
        valid_length = i + 1
        return np.stack((vec1d, heads)), valid_length


@register_vectorizer(name="dict-wordpiece1d-with-heads")
class DictWordPieceVectoizer1DWithHeads(WordPieceVectorizer1DWithHeads):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '~~')

    def iterable(self, tokens):
        return super().iterable(_token_iterator(self, tokens))


@register_vectorizer(name='wordpiece1d-with-heads-dict')
class WordPieceVectorizer1DWithHeadsDict(WordPieceVectorizer1D):
    """This vectorizer produces the tokens and the index of the heads at the same time.

    In this one the results are output as a dict
    """

    def iterable(self, tokens):
        yield '[CLS]', 0
        for tok in tokens:
            if tok == '<unk>':
                yield '[UNK]', 1
            elif tok == '<EOS>':
                yield '[SEP]', 0
            else:
                for i, subtok in enumerate(self.tokenizer.tokenize(tok)):
                    if i == 0:
                        yield subtok, 1
                    else:
                        yield subtok, 0
        yield '[SEP]', 0

    def _next_element(self, tokens, vocab):
        for atom, head in self.iterable(tokens):
            value = vocab.get(atom, vocab['[UNK]'])
            yield value, head

    def run(self, tokens, vocab):
        self.mxlen = self.max_seen if self.mxlen < 0 else self.mxlen
        vec1d = np.zeros(self.mxlen, dtype=np.long)
        heads = np.zeros(self.mxlen, dtype=np.long)
        for i, (atom, head) in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
            heads[i] = head
        heads = mask_to_index(heads)
        valid_length = i + 1
        return {'tokens': vec1d, 'heads': heads}, valid_length


@register_vectorizer(name="dict-wordpiece1d-with-heads-dict")
class DictWordPieceVectoizer1DWithHeadsDict(WordPieceVectorizer1DWithHeadsDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '~~')

    def iterable(self, tokens):
        return super().iterable(_token_iterator(self, tokens))


@register_vectorizer(name='heads-wordpiece1d')
class HeadedWordPieceVectorizer1D(WordPieceVectorizer1D):
    """Output a list of indices that are the heads of the subword tokens."""

    def iterable(self, tokens):
        j = 1  # Don't point at [CLS]
        for tok in tokens:
            # Unk is something we want to label
            if tok == '<unk>':
                yield j
            # We don't want to label [SEP]
            elif tok == '<EOS>':
                pass
            else:
                for i, subtok in enumerate(self.tokenizer.tokenize(tok)):
                    if i == 0:
                        yield j  # Head
                    else:
                        j += 1
            j += 1

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            yield atom

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

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        # Count the mxlen of the actual text otherwise we are to short
        # If this was a mask we wouldn't have this issue
        for tok in super().iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter


@register_vectorizer(name='dict-heads-wordpiece1d')
class DictHeadedWordPieceVectorizer1D(HeadedWordPieceVectorizer1D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def iterable(self, tokens):
        return super().iterable(_token_iterator(self, tokens))

    def count(self, tokens):
        return super().count(_token_iterator(self, tokens))


class LabelHeadedWordPieceVectorizer1D(DictHeadedWordPieceVectorizer1D):
    """Convert labels so just the heads have labels, other labels are PAD"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', ['text', 'y']))
        self.delim = kwargs.get('token_delim', '~~')

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            # Don't include the pad in the output because it is already in the vocab
            if tok != Offsets.VALUES[Offsets.PAD]:
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

    def iterable(self, tokens):
        yield Offsets.VALUES[Offsets.PAD]
        for tok in _token_iterator(self, tokens):
            text, label = tok.split(self.delim)
            if text == '<unk>':
                yield label
            elif text == '<EOS>':
                yield label
            else:
                for i, subtok in enumerate(self.tokenizer.tokenize(text)):
                    if i == 0:
                        yield label
                        continue
                    yield Offsets.VALUES[Offsets.PAD]
        yield Offsets.VALUES[Offsets.PAD]

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
        # This is the actual number of tags, aka the number of tokens before the wordpiece
        valid_length = np.sum(vec1d != Offsets.PAD)
        return vec1d, valid_length


@register_reader(task='tagger', name='bert')
class BERTCONLLReader(CONLLSeqReader):
    def __init__(self, vectorizers, trim=False, truncate=False, mxlen=-1, **kwargs):
        super().__init__(vectorizers, trim, truncate, mxlen, **kwargs)
        self.named_fields = kwargs.get('named_fields', {})
        self.label_vectorizer = LabelHeadedWordPieceVectorizer1D(fields=['text', 'y'], mxlen=mxlen)


class BERTBaseEmbeddings(PyTorchEmbeddings):

    def __init__(self, name, **kwargs):
        super(BERTBaseEmbeddings, self).__init__(name=name, **kwargs)
        global BERT_TOKENIZER
        self.dsz = kwargs.get('dsz')
        if BERT_TOKENIZER is None:
            BERT_TOKENIZER = BertTokenizer.from_pretrained(kwargs.get('embed_file'))
        self.model = BertModel.from_pretrained(kwargs.get('embed_file'))
        self.vocab = BERT_TOKENIZER.vocab
        self.vsz = len(BERT_TOKENIZER.vocab)  # 30522 self.model.embeddings.word_embeddings.num_embeddings

    def get_vocab(self):
        return self.vocab

    def get_dsz(self):
        return self.dsz

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("bert", **kwargs)
        c.checkpoint = embeddings
        return c

    def forward(self, x):

        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1)
        input_type_ids = torch.zeros(x.shape, device=x.device, dtype=torch.long)
        all_layers, pooled = self.model(x, token_type_ids=input_type_ids, attention_mask=input_mask)
        z = self.get_output(all_layers, pooled)
        return z

    def get_output(self, all_layers, pooled):
        pass


@register_embeddings(name='bert')
class BERTEmbeddings(BERTBaseEmbeddings):
    """BERT sequence embeddings, used for a feature-ful representation of finetuning sequence tasks.

    If operator == 'concat' result is [B, T, #Layers * H] other size the layers are meaned and the shape is [B, T, H]
    """

    def __init__(self, name, **kwargs):
        """BERT sequence embeddings, used for a feature-ful representation of finetuning sequence tasks.

        If operator == 'concat' result is [B, T, #Layers * H] other size the layers are meaned and the shape is [B, T, H]
        """
        super(BERTEmbeddings, self).__init__(name=name, **kwargs)
        self.layer_indices = kwargs.get('layers', [-1, -2, -3, -4])
        self.operator = kwargs.get('operator', 'concat')
        self.finetune = kwargs.get('finetune', False)

    def get_output(self, all_layers, pooled):
        if self.finetune:
            layers = [all_layers[layer_index] for layer_index in self.layer_indices]
        else:
            layers = [all_layers[layer_index].detach() for layer_index in self.layer_indices]
        if self.operator != 'concat':
            z = torch.cat([l.unsqueeze(-1) for l in layers], dim=-1)
            z = torch.mean(z, dim=-1)
        else:
            z = torch.cat(layers, dim=-1)
        return z

    def extra_repr(self):
        return f"finetune={self.finetune}, combination={self.operator}, layers={self.layer_indices}"


@register_embeddings(name='bert-pooled')
class BERTPooledEmbeddings(BERTBaseEmbeddings):

    def __init__(self, name, **kwargs):
        super(BERTPooledEmbeddings, self).__init__(name=name, **kwargs)

    def get_output(self, all_layers, pooled):
        return pooled


class IdentityTransducer(nn.Module):
    """A module that just passes tensor but also has an output shape."""
    def __init__(self, input_sz):
        super().__init__()
        self.output_dim = input_sz

    def forward(self, inputs):
        inputs, _ = tensor_and_lengths(inputs)
        return inputs


def select_heads(outputs, head_idx, dim=1):
    """Select the heads of wordpiece spans.

    :param outputs: The Bert logits, Tensor[B, T, H]
    :param head_idx: The head indices, Tensor[B, T2]
    :param dim: The Time dimension.
    :returns: The logits for the heads, Tensor[B, T2, H]
    """
    idx = head_idx.unsqueeze(-1).expand(list(head_idx.shape) + [outputs.size(-1)])
    return torch.gather(outputs, dim, idx)


@register_model(task='tagger', name='bert')
class BertTaggerModel(TaggerModelBase):

    def init_encoder(self, input_sz, **kwargs):
        return IdentityTransducer(input_sz)

    def make_input(self, batch_dict):

        example_dict = dict({})
        lengths = torch.from_numpy(batch_dict[self.lengths_key])
        lengths, perm_idx = lengths.sort(0, descending=True)

        if self.gpu:
            lengths = lengths.cuda()
        example_dict['lengths'] = lengths
        # The vectorizer outputs a [2, T] tensor which is stacked into [B, 2, T].
        # The first index in the 2 dim is the tokens and the second is the heads
        for key in self.embeddings.keys():
            example_dict[key] = self.input_tensor(key, batch_dict, perm_idx)

        # This assumes all heads are aligned
        heads = [k for k in batch_dict.keys() if k.endswith('_heads')][0]
        example_dict['heads'] = torch.from_numpy(batch_dict[heads])[perm_idx]
        if self.gpu:
            example_dict['heads'] = example_dict['heads'].cuda()


        y = batch_dict.get('y')
        if y is not None:
            y = torch.from_numpy(y)[perm_idx]
            if self.gpu:
                y = y.cuda()
            # We generated the tags with gaps so non-heads have a value of Offsets.PAD
            # Here we select only the tags for heads. This results in a tag seq so the
            # beginning of the list is the tags for the heads and the pad tags are moved
            # to the back. This lets use use the normal evaluation code
            y = select_heads(y.unsqueeze(-1), example_dict['heads']).squeeze(-1)
            example_dict['y'] = y

        # Add y_lengths to the dict so it gets sorted correctly
        y_lengths = batch_dict.get('y_lengths')
        if y_lengths is not None:
            y_lengths = torch.from_numpy(y_lengths)[perm_idx]
            example_dict['y_lengths'] = y_lengths

        ids = batch_dict.get('ids')
        if ids is not None:
            ids = torch.from_numpy(ids)[perm_idx]
            if self.gpu:
                ids = ids.cuda()
            example_dict['ids'] = ids

        return example_dict

    def compute_loss(self, inputs):
        tags = inputs['y']
        lengths = inputs['lengths']
        unaries = self.layers.transduce(inputs)
        # Select only the heads of the subword spans
        unaries = select_heads(unaries, inputs['heads'])
        assert unaries.shape[:2] == tags.shape
        return self.layers.neg_log_loss(unaries, tags, lengths)

    def input_tensor(self, key, batch_dict, perm_idx):
        tensor = torch.from_numpy(batch_dict[key])
        tensor = self.drop_inputs(key, tensor)
        tensor = tensor[perm_idx]
        if self.gpu:
            tensor = tensor.cuda()
        return tensor

    def forward(self, input):
        unaries = self.layers.transduce(input)
        unaries = select_heads(unaries, input['heads'])
        # This is the same calculation as the y_heads tensor
        lengths = torch.sum(input['heads'] != 0, dim=1)
        longest = torch.max(lengths)
        unaries = unaries[:, :longest, ...]
        return self.layers.decode(unaries, lengths)


@register_model(task='tagger', name='bert2')
class BertTaggerVectWithHeads(BertTaggerModel):
    """This is a version of the tagger where the token values and heads are computed together and separated in the make_intput
    """

    def make_input(self, batch_dict):

        example_dict = dict({})
        lengths = torch.from_numpy(batch_dict[self.lengths_key])
        lengths, perm_idx = lengths.sort(0, descending=True)

        if self.gpu:
            lengths = lengths.cuda()
        example_dict['lengths'] = lengths
        # The vectorizer outputs a [2, T] tensor which is stacked into [B, 2, T].
        # The first index in the 2 dim is the tokens and the second is the heads
        for key in self.embeddings.keys():
            batch_dict[f'{key}_heads'] = batch_dict[key][:, 1, ...]
            batch_dict[key] = batch_dict[key][:, 0, ...]
            example_dict[key] = self.input_tensor(key, batch_dict, perm_idx)

        # This assumes all heads are aligned
        heads = [k for k in batch_dict.keys() if k.endswith('_heads')][0]
        example_dict['heads'] = torch.from_numpy(batch_dict[heads])[perm_idx]
        if self.gpu:
            example_dict['heads'] = example_dict['heads'].cuda()


        y = batch_dict.get('y')
        if y is not None:
            y = torch.from_numpy(y)[perm_idx]
            if self.gpu:
                y = y.cuda()
            # We generated the tags with gaps so non-heads have a value of Offsets.PAD
            # Here we select only the tags for heads. This results in a tag seq so the
            # beginning of the list is the tags for the heads and the pad tags are moved
            # to the back. This lets use use the normal evaluation code
            y = select_heads(y.unsqueeze(-1), example_dict['heads']).squeeze(-1)
            example_dict['y'] = y

        # Add y_lengths to the dict so it gets sorted correctly
        y_lengths = batch_dict.get('y_lengths')
        if y_lengths is not None:
            y_lengths = torch.from_numpy(y_lengths)[perm_idx]
            example_dict['y_lengths'] = y_lengths

        ids = batch_dict.get('ids')
        if ids is not None:
            ids = torch.from_numpy(ids)[perm_idx]
            if self.gpu:
                ids = ids.cuda()
            example_dict['ids'] = ids

        return example_dict


@register_model(task='tagger', name='bert-dict')
class BertTaggerVectWithHeadsDict(BertTaggerModel):
    """This is a version of the tagger where the token values and heads are computed together and separated in the make_intput this was a first attempt and ended up bing slow af
    """

    def make_input(self, batch_dict):

        example_dict = dict({})
        lengths = torch.from_numpy(batch_dict[self.lengths_key])
        lengths, perm_idx = lengths.sort(0, descending=True)

        if self.gpu:
            lengths = lengths.cuda()
        example_dict['lengths'] = lengths
        # The batcher auto stacks the datafeed for each key, because we have a dict
        # as the value it gets stacked into a array of objects. Here we separate them
        for key in self.embeddings.keys():
            if batch_dict[key].dtype == np.dtype('O'):
                batch_dict[f'{key}_heads'] = np.stack([x['heads'] for x in batch_dict[key]])
                batch_dict[key] = np.stack([x['tokens'] for x in batch_dict[key]])
            example_dict[key] = self.input_tensor(key, batch_dict, perm_idx)

        # This assumes all heads are aligned
        heads = [k for k in batch_dict.keys() if k.endswith('_heads')][0]
        example_dict['heads'] = torch.from_numpy(batch_dict[heads])[perm_idx]
        if self.gpu:
            example_dict['heads'] = example_dict['heads'].cuda()

        y = batch_dict.get('y')
        if y is not None:
            y = torch.from_numpy(y)[perm_idx]
            if self.gpu:
                y = y.cuda()
            # We generated the tags with gaps so non-heads have a value of Offsets.PAD
            # Here we select only the tags for heads. This results in a tag seq so the
            # beginning of the list is the tags for the heads and the pad tags are moved
            # to the back. This lets use use the normal evaluation code
            y = select_heads(y.unsqueeze(-1), example_dict['heads']).squeeze(-1)
            example_dict['y'] = y

        # Add y_lengths to the dict so it gets sorted correctly
        y_lengths = batch_dict.get('y_lengths')
        if y_lengths is not None:
            y_lengths = torch.from_numpy(y_lengths)[perm_idx]
            example_dict['y_lengths'] = y_lengths

        ids = batch_dict.get('ids')
        if ids is not None:
            ids = torch.from_numpy(ids)[perm_idx]
            if self.gpu:
                ids = ids.cuda()
            example_dict['ids'] = ids

        return example_dict
