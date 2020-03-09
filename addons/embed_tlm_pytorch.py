from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import Counter


from eight_mile.utils import read_json
from eight_mile.pytorch.layers import TransformerEncoderStack, subsequent_mask
from eight_mile.pytorch.embeddings import PyTorchEmbeddings, PositionalLookupTableEmbeddings, LearnedPositionalLookupTableEmbeddings
from baseline.embeddings import register_embeddings
from baseline.pytorch.embeddings import PyTorchEmbeddingsModel
from baseline.vectorizers import register_vectorizer, AbstractVectorizer, BPEVectorizer1D
from baseline.pytorch.torchy import *
from eight_mile.pytorch.serialize import load_tlm_npz
import torch.nn as nn

@register_vectorizer(name='tlm-wordpiece')
class WordPieceVectorizer1D(AbstractVectorizer):
    """Define a Baseline Vectorizer that can do WordPiece with BERT tokenizer

    If you use tokens=subword, this vectorizer is used, and so then there is
    a dependency on bert_pretrained_pytorch
    """

    def __init__(self, **kwargs):
        """Loads a BertTokenizer using bert_pretrained_pytorch

        :param kwargs:
        """
        super().__init__(kwargs.get('transform_fn'))
        from pytorch_pretrained_bert import BertTokenizer
        self.max_seen = 128
        handle = kwargs.get('embed_file')
        self.tokenizer = BertTokenizer.from_pretrained(handle, do_lower_case=False)
        self.mxlen = kwargs.get('mxlen', -1)

    def count(self, tokens):
        seen = 0
        counter = Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def iterable(self, tokens):
        for tok in tokens:
            if tok == '<unk>':
                yield '[UNK]'
            elif tok == '<EOS>':
                yield '[SEP]'
            else:
                for subtok in self.tokenizer.tokenize(tok):
                    yield subtok
        yield '[CLS]'

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
                vec1d[i] = vocab.get('[CLS]')
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


@register_vectorizer(name='tlm-bpe')
class BPEVectorizer1DFT(BPEVectorizer1D):
    """Override bpe1d to geneate [CLS] """
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
        yield '[CLS]'

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = np.zeros(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                vec1d[i] = vocab.get('[CLS]')
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length


class TransformerLMEmbeddings(PyTorchEmbeddings):
    """Support embeddings trained with the TransformerLanguageModel class

    This method supports either subword or word embeddings, not characters

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.vocab = read_json(kwargs.get('vocab_file'))
        self.cls_index = self.vocab['[CLS]']
        self.vsz = len(self.vocab)
        layers = int(kwargs.get('layers', 18))
        num_heads = int(kwargs.get('num_heads', 10))
        pdrop = kwargs.get('dropout', 0.1)
        self.d_model = int(kwargs.get('dsz', kwargs.get('d_model', 410)))
        d_ff = int(kwargs.get('d_ff', 2100))
        d_k = kwargs.get('d_k')
        rpr_k = kwargs.get('rpr_k')
        embed_type = kwargs.get('word_embed_type', 'positional')
        if embed_type == 'positional':
            x_embedding = PositionalLookupTableEmbeddings(vsz=self.vsz, dsz=self.d_model)
        elif embed_type == 'learned-positional':
            x_embedding = LearnedPositionalLookupTableEmbeddings(vsz=self.vsz, dsz=self.d_model)
        self.dsz = self.init_embed({'x': x_embedding})
        self.proj_to_dsz = pytorch_linear(self.dsz, self.d_model) if self.dsz != self.d_model else _identity
        self.transformer = TransformerEncoderStack(num_heads, d_model=self.d_model, pdrop=pdrop, scale=True, layers=layers, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k)
        self.mlm = kwargs.get('mlm', False)
        self.finetune = kwargs.get('finetune', True)

    def embed(self, input):
        embedded = self.embeddings['x'](input)
        embedded_dropout = self.embed_dropout(embedded)
        if self.embeddings_proj:
            embedded_dropout = self.embeddings_proj(embedded_dropout)
        return embedded_dropout

    def init_embed(self, embeddings, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.1))
        self.embed_dropout = nn.Dropout(pdrop)
        self.embeddings = EmbeddingsStack(embeddings)
        input_sz = 0
        for k, embedding in embeddings.items():
            input_sz += embedding.get_dsz()

        projsz = kwargs.get('projsz')
        if projsz:
            self.embeddings_proj = pytorch_linear(input_sz, projsz)
            print('Applying a transform from {} to {}'.format(input_sz, projsz))
            return projsz
        else:
            self.embeddings_proj = None
        return input_sz

    def _model_mask(self, nctx):
        """This function creates the mask that controls which token to be attended to depending on the model. A causal
        LM should have a subsequent mask; and a masked LM should have no mask."""
        if self.mlm:
            return torch.ones((1, 1, nctx, nctx), dtype=torch.long)
        else:
            return subsequent_mask(nctx)

    def forward(self, x):
        # the following line masks out the attention to padding tokens
        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1).unsqueeze(1).unsqueeze(1)
        # the following line builds mask depending on whether it is a causal lm or masked lm
        input_mask = input_mask & self._model_mask(x.shape[1]).type_as(input_mask)
        embedding = self.embed(x)
        embedding = self.proj_to_dsz(embedding)
        transformer_out = self.transformer((embedding, input_mask))
        z = self.get_output(x, transformer_out)
        return z

    def get_output(self, inputs, z):
        return z if self.finetune else z.detach()

    def get_vocab(self):
        return self.vocab

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.d_model

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("tlm-words-embed", **kwargs)
        if embeddings.endswith('.pth'):
            unmatch = c.load_state_dict(torch.load(embeddings), strict=False)
            if unmatch.missing_keys or len(unmatch.unexpected_keys) > 2:
                print("Warning: Embedding doesn't match with the checkpoint being loaded.")
                print(f"missing keys: {unmatch.missing_keys}\n unexpected keys: {unmatch.unexpected_keys}")
        elif embeddings.endswith('.npz'):
            load_tlm_npz(c, embeddings)
        return c


@register_embeddings(name='tlm-words-embed')
class TransformerLMEmbeddingsModel(PyTorchEmbeddingsModel, TransformerLMEmbeddings):
    """Register embedding model for usage in mead"""
    pass


def _identity(x):
    return x


def _mean_pool(_, embeddings):
    return torch.mean(embeddings, 1, False)


def _max_pool(_, embeddings):
    return torch.max(embeddings, 1, False)[0]



@register_embeddings(name='tlm-words-embed-pooled')
class TransformerLMPooledEmbeddingsModel(TransformerLMEmbeddingsModel):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        pooling = kwargs.get('pooling', 'cls')
        if pooling == 'max':
            self.pooling_op = _max_pool
        elif pooling == 'mean':
            self.pooling_op = _mean_pool
        elif pooling == 'sqrt_length':
            self.pooling_op = self._sqrt_length_pool
        else:
            self.pooling_op = self._cls_pool

    def _sqrt_length_pool(self, inputs, embeddings):
        lengths = (inputs != 0).sum(1)
        sqrt_length = lengths.float().sqrt().unsqueeze(1)
        embeddings = embeddings.sum(1) / sqrt_length
        return embeddings

    def _cls_pool(self, inputs, tensor):
        pooled = tensor[inputs == self.cls_index]
        return pooled

    def get_output(self, inputs, z):
        z = self.pooling_op(inputs, z)
        return z if self.finetune else z.detach()
