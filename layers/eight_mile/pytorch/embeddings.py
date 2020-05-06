import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from eight_mile.utils import Offsets, is_sequence, calc_nfeats
from eight_mile.pytorch.layers import *


class PyTorchEmbeddings(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    @property
    def output_dim(self):
        return self.get_dsz()

    def encode(self, x):
        return self(x)


class LookupTableEmbeddings(PyTorchEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vsz = kwargs.get("vsz")
        self.dsz = kwargs.get("dsz")
        self.finetune = kwargs.get("finetune", True)
        self.dropin = kwargs.get("dropin", 0.0)

        weights = kwargs.get("weights")
        if weights is None:
            self.embeddings = nn.Embedding(self.vsz, self.dsz, padding_idx=Offsets.PAD)
        else:
            self.embeddings = pytorch_embedding(weights, self.finetune)
            # This makes sure that if you init with a weight and not vsz it will still be available
            self.vsz, self.dsz = weights.shape

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.dsz

    def forward(self, x):
        if not self.dropin:
            return self.embeddings(x)

        mask = self.embeddings.weight.data.new().resize_((self.embeddings.weight.size(0),
                                                          1)).bernoulli_(1 - self.dropin).expand_as(self.embeddings.weight) / (1 - self.dropin)
        masked_embed_weight = mask * self.embeddings.weight
        output = torch.nn.functional.embedding(x, masked_embed_weight,
                                               self.embeddings.padding_idx, self.embeddings.max_norm, self.embeddings.norm_type,
                                               self.embeddings.scale_grad_by_freq, self.embeddings.sparse)
        return output

    def extra_repr(self):
        return f"finetune=False" if not self.finetune else ""


class CharConvEmbeddings(PyTorchEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.nfeat_factor = kwargs.get("nfeat_factor")
        self.cfiltsz = kwargs.get("cfiltsz", kwargs.get("filtsz", [3]))
        self.max_feat = kwargs.get("max_feat", 30)
        self.gating = kwargs.get("gating", "skip")
        self.num_gates = kwargs.get("num_gates", 1)
        self.activation = kwargs.get("activation", "tanh")
        self.wsz = kwargs.get("wsz", 30)
        self.projsz = kwargs.get("projsz", 0)
        self.pdrop = kwargs.get("pdrop", 0.5)
        self.filtsz, self.nfeats = calc_nfeats(self.cfiltsz, self.nfeat_factor, self.max_feat, self.wsz)
        self.conv_outsz = int(np.sum(self.nfeats))
        self.outsz = self.conv_outsz
        if self.projsz > 0:
            self.outsz = self.projsz
        self.proj = pytorch_linear(self.conv_outsz, self.outsz)

        self.embeddings = LookupTableEmbeddings(**kwargs)
        self.char_comp = WithDropout(
            ParallelConv(self.embeddings.output_dim, self.nfeats, self.filtsz, self.activation), self.pdrop
        )

        GatingConnection = SkipConnection if self.gating == "skip" else Highway
        self.gating_seq = nn.Sequential(
            OrderedDict(
                [("gate-{}".format(i), GatingConnection(self.char_comp.output_dim)) for i in range(self.num_gates)]
            )
        )

    def get_dsz(self):
        return self.outsz

    def get_vsz(self):
        return self.vsz

    def forward(self, xch):

        # For starters we need to perform embeddings for each character
        # (TxB) x W -> (TxB) x W x D
        _0, _1, W = xch.shape
        char_vecs = self.embeddings(xch.view(-1, W))
        # (TxB) x D x W
        # char_vecs = char_embeds.transpose(1, 2).contiguous()

        #        pytorch_activation(self.activation_type)
        mots = self.char_comp(char_vecs)
        gated = self.gating_seq(mots)
        if self.projsz:
            gated = self.proj(gated)
        return gated.view(_0, _1, self.get_dsz())


class CharLSTMEmbeddings(PyTorchEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embed = LookupTableEmbeddings(**kwargs)
        self.lstmsz = kwargs.get("lstmsz", 50)
        layers = kwargs.get("layers", 1)
        pdrop = kwargs.get("pdrop", 0.5)
        unif = kwargs.get("unif", 0)
        weight_init = kwargs.get("weight_init", "uniform")
        self.char_comp = BiLSTMEncoderHidden(
            self.embed.output_dim, self.lstmsz, layers, pdrop, unif=unif, initializer=weight_init
        )

    def forward(self, xch):
        B, T, W = xch.shape
        flat_chars = xch.view(-1, W)
        char_embeds = self.embed(flat_chars)

        # Calculate the lengths of each word
        lengths = torch.sum(flat_chars != Offsets.PAD, dim=1)

        # Sort the input to appease the cuDNN gods
        sorted_word_lengths, perm_idx = lengths.sort(0, descending=True)
        sorted_feats = char_embeds[perm_idx].transpose(0, 1).contiguous()

        # cuDNN throws an error if there is an input with a length of 0, this happens when the "word"
        # is actually a "<PAD>" so there are no characters to run the LSTM over. Here we just say
        # that the lengths is 1. This will make cudnn happy and we will just get junk in that spot
        patched_lengths = sorted_word_lengths.masked_fill(sorted_word_lengths == 0, 1)

        # Run the LSTM
        hidden = self.char_comp((sorted_feats, patched_lengths))

        # Create a mask that is true when the sorted length is 0 (where the word was a pad) so that
        # we can mask out the junk that the lstm created because we needed a length of 1
        hidden = hidden.masked_fill((sorted_word_lengths == 0).unsqueeze(-1), 0)

        # Undo the sort so that the representations of the words are in the correct part of the sentence.
        results = unsort_batch(hidden, perm_idx)

        return results.reshape((B, T, -1))

    def get_dsz(self):
        return self.lstmsz

    def get_vsz(self):
        return self.embed.get_vsz()


class CharTransformerEmbeddings(PyTorchEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embed = LookupTableEmbeddings(**kwargs)
        self.d_model = kwargs.get("wsz", 30)
        self.num_heads = kwargs.get("num_heads", 3)
        self.rpr_k = kwargs.get("rpr_k", 10)
        layers = kwargs.get("layers", 1)
        pdrop = kwargs.get("pdrop", 0.5)
        self.char_comp = TransformerEncoderStackWithLengths(
            self.num_heads, self.d_model, pdrop, False, layers, rpr_k=self.rpr_k, input_sz=self.embed.output_dim
        )

    def forward(self, xch):
        B, T, W = xch.shape
        flat_chars = xch.view(-1, W)
        char_embeds = self.embed(flat_chars)

        # Calculate the lengths of each word
        lengths = torch.sum(flat_chars != Offsets.PAD, dim=1)
        results = self.char_comp((char_embeds, lengths))
        # B,T,H output, how to pool this
        pooled = torch.max(results, -2, keepdims=False)[0]
        return pooled.reshape((B, T, -1))

    def get_dsz(self):
        return self.d_model

    def get_vsz(self):
        return self.embed.get_vsz()


class PositionalMixin(nn.Module):
    """A Mixin that provides functionality to generate positional embeddings to be added to the normal embeddings.

    Note, mixins need to be before the base case when used, i.e.
        `Embedding(Mixin, BaseEmbed)` NOT `Embedding(BaseEmbed, Mixin)`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def positional(self, length):
        pass

    def extra_repr(self):
        return f"mxlen={self.mxlen}"


class SinusoidalPositionalMixin(PositionalMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # This could get us in trouble, if in doubt, pick something big
        self.mxlen = kwargs.get("mxlen", 1000)
        max_timescale = kwargs.get("max_timescale", 1.0e4)

        word_dsz = self.get_dsz()

        log_timescale_increment = math.log(max_timescale) / word_dsz
        inv_timescales = torch.exp(torch.arange(0, word_dsz, 2).float() * -log_timescale_increment)

        pe = torch.zeros(self.mxlen, word_dsz)
        position = torch.arange(0, self.mxlen).float().unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * inv_timescales)
        pe[:, 1::2] = torch.cos(position * inv_timescales)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def positional(self, length):
        return self.pe[:, :length]


class LearnedPositionalMixin(PositionalMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mxlen = int(kwargs.get("mxlen", 512))
        self.pos_embeddings = nn.Embedding(self.mxlen, self.get_dsz())

    def positional(self, length):
        return self.pos_embeddings(
            torch.arange(length, dtype=torch.long, device=self.pos_embeddings.weight.device)
        ).unsqueeze(0)


class BERTLookupTableEmbeddings(LookupTableEmbeddings):
    """
    BERT style embeddings with a 0 token type

    TODO: Get rid of this, we dont need it anymore
    If you want to use BERT with token types, make a `LearnedPositionalLookupTableEmbeddings` feature
    and a `LookupTableEmbeddings` feature (for the token type)
    and put them in an `EmbeddingsStack` with an embeddings_reduction='sum-layer-norm' on the model

    Otherwise, if you do not plan on setting the token type, use the `LearnedPositionalLookupTableEmbeddingsWithBias`,
    which will add the BERT token_type=0 weights into the pos + word_embed and is more efficient
    than this class, since it doesnt do any memory allocation on the fly
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        self.mxlen = int(kwargs.get('mxlen', 512))
        self.tok_type_vsz = kwargs['tok_type_vsz']
        self.pos_embeddings = nn.Embedding(self.mxlen, self.get_dsz())
        self.tok_embeddings = nn.Embedding(self.tok_type_vsz, self.get_dsz())
        self.ln = nn.LayerNorm(self.get_dsz(), eps=1e-12)

    def forward(self, x):
        zeros = torch.zeros_like(x)
        x = super().forward(x)
        x = x + self.positional(x.size(1)) + self.tok_embeddings(zeros)
        x = self.ln(x)
        return self.dropout(x)

    def positional(self, length):
        return self.pos_embeddings(
            torch.arange(length, dtype=torch.long, device=self.pos_embeddings.weight.device)
        ).unsqueeze(0)


class LearnedPositionalLookupTableEmbeddingsWithBias(LookupTableEmbeddings):
    """Learned positional lookup table embeddings wih a bias and layer norm

    This is just a typical learned positional embedding but with a learnable
    bias and a layer norm.  This is equivalent to BERT embeddings when the
    token_type is not set.

    If you are using BERT but you have no interest in using token type embeddings
    (IOW if you are setting all the values of that feature zero anyhow), using this
    object is faster and simpler than having a separate vectorizer for token type.

    If you have a need for token type embeddings, you will want to create 2 sets of embeddings,
    one that acts on the tokens, of type `LearnedPositionalLookupTableEmbeddings` and one of the type
    `LookupTableEmbeddings` for the token type feature

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))
        self.mxlen = int(kwargs.get('mxlen', 512))
        self.pos_embeddings = nn.Embedding(self.mxlen, self.get_dsz())
        self.bias = nn.Parameter(torch.zeros(self.get_dsz()))

    def forward(self, x):
        x = super().forward(x)
        x = x + self.positional(x.size(1)) + self.bias
        return x

    def positional(self, length):
        return self.pos_embeddings(
            torch.arange(length, dtype=torch.long, device=self.pos_embeddings.weight.device)
        ).unsqueeze(0)


class PositionalLookupTableEmbeddings(SinusoidalPositionalMixin, LookupTableEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))
        self.scale = math.sqrt(self.get_dsz())

    def forward(self, x):
        """Add a positional encoding to the embedding, followed by dropout

        :param x: The temporal signal in, to which the positional embeddings are applied
        :return: Embedded output
        """
        x = super().forward(x) * self.scale
        x = x + self.positional(x.size(1))
        return self.dropout(x)


class LearnedPositionalLookupTableEmbeddings(LearnedPositionalMixin, LookupTableEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def forward(self, x):
        T = x.size(1)
        x = super().forward(x)
        pos = self.positional(T)
        return self.dropout(x + pos)


class PositionalCharConvEmbeddings(SinusoidalPositionalMixin, CharConvEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))
        self.scale = math.sqrt(self.get_dsz())

    def forward(self, xch):
        """Add a positional encoding to the embedding, followed by dropout

        :param xch: The temporal signal in, to which the positional embeddings are applied
        :return: Embedded output
        """
        xch = super().forward(xch) * self.scale
        xch = xch + self.positional(xch.size(1))
        return self.dropout(xch)


class LearnedPositionalCharConvEmbeddings(LearnedPositionalMixin, CharConvEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def forward(self, xch):
        """Add a positional encoding to the embedding, followed by dropout

        :param xch: The temporal signal in, to which the positional embeddings are applied
        :return: Embedded output
        """
        xch = super().forward(xch)
        xch = xch + self.positional(xch.size(1))
        return self.dropout(xch)


class PositionalCharLSTMEmbeddings(SinusoidalPositionalMixin, CharLSTMEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))
        self.scale = math.sqrt(self.get_dsz())

    def forward(self, xch):
        xch = super().forward(xch) * self.scale
        xch = xch + self.positional(xch.size(1))
        return self.dropout(xch)


class LearnedPositionalCharLSTMEmbeddings(LearnedPositionalMixin, CharLSTMEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def forward(self, xch):
        xch = super().forward(xch)
        xch = xch + self.positional(xch.size(1))
        return self.dropout(xch)
