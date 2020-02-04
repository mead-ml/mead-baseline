"""
Hierarchically-Refined Label Attention Network for Sequence Labeling from here https://arxiv.org/pdf/1908.08676.pdf

Couple notes of possible difference with the paper because it isn't clear

1. In section 4.2 when discussing Using multihead attention to compute H^t they have this equation

    H^l = concat(head, ..., head_k) + H^w

   This seems to suggest they have a residual connection around the multi head attention, this seems weird given that at
   the next step they also concatenate with H^w. Their code doesn't do the addition suggested by their paper
   https://github.com/Nealcly/BiLSTM-LAN/blob/082fb6aec69b468bcfb0bff5aeaa2e43f4073965/model/lstm_attention.py#L24

2. The paper doesn't say that the last layer only has a single attention head but it makes sense, other wise you would only get
   to assign the probability to a subset of labels given a subset of the features. This is what they do in their code though.
   https://github.com/Nealcly/BiLSTM-LAN/blob/082fb6aec69b468bcfb0bff5aeaa2e43f4073965/model/wordsequence.py#L44

3. In their code they have a query masking step which seems pointless? You don't need to mask the padded inputs because
   their attentions are calculated separately from each other so even though they will calculate junk as long as the loss
   ignores them it doesn't matter.
"""

import math
import torch
import torch.nn as nn
from eight_mile.pytorch.embeddings import LookupTableEmbeddings
from eight_mile.pytorch.layers import (
    BiLSTMEncoderSequence,
    MultiHeadedAttention,
    tensor_and_lengths,
    bth2tbh,
    tbh2bth,
    SequenceSequenceAttention,
    Dense,
)
from baseline.model import register_model
from baseline.pytorch.tagger.model import TaggerModelBase


class BiLSTMLANEncoder(nn.Module):
    def __init__(self, insz, hsz, nlayers, nlabels, num_heads=4, **kwargs):
        super().__init__()
        assert nlayers > 1, "You need at least 2 layers for a BiLSTMLANEncoder"
        blstm_dropout = kwargs.get('blstm_dropout', 0.5)

        kwargs['batch_first'] = False
        blstms = [BiLSTMEncoderSequence(insz, hsz, 1, pdrop=blstm_dropout, **kwargs)]
        for _ in range(nlayers - 1):
            blstms.append(BiLSTMEncoderSequence(hsz * 2, hsz, 1, pdrop=blstm_dropout, **kwargs))
        self.blstms = nn.ModuleList(blstms)

        mha_dropout = kwargs.get('mha_dropout', 0.1)
        mhas = [
            MultiHeadedAttention(num_heads=num_heads, d_model=hsz, dropout=mha_dropout, scale=True)
            for _ in range(nlayers - 1)
        ]
        mhas.append(TruncatedMultiHeadedAttention(num_heads=1, d_model=hsz, dropout=0.0, scale=True))
        self.mhas = nn.ModuleList(mhas)

        self.label_embed = LookupTableEmbeddings(vsz=nlabels, dsz=hsz)
        self.nlabels = nlabels

    @property
    def output_dim(self):
        return self.nlabels

    def forward(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        batchsz = inputs.size(1)
        labels = self.label_embed(torch.arange(self.nlabels, device=inputs.device).unsqueeze(0))
        labels = labels.expand([batchsz] + list(labels.shape[1:]))
        for i in range(len(self.blstms) - 1):
            out = self.blstms[i]((inputs, lengths))
            out = tbh2bth(out)
            # No mask because we can attend to any label
            label_attn = self.mhas[i]((out, labels, labels, None))
            inputs = torch.cat([out, label_attn], dim=2)
            inputs = bth2tbh(inputs)
        out = self.blstms[-1]((inputs, lengths))
        out = tbh2bth(out)
        attn_weights = self.mhas[-1]((out, labels, labels, None))
        attn_weights = bth2tbh(attn_weights)
        return attn_weights


@register_model(task='tagger', name='blstm-lan')
class BLSTMLANTaggerModel(TaggerModelBase):
    def init_encoder(self, input_sz, **kwargs):
        nlayers = int(kwargs.get('layers', 2))
        blstm_dropout = float(kwargs.get('blstm_dropout', 0.5))
        mha_dropout = float(kwargs.get('mha_dropout', 0.1))
        unif = kwargs.get('unif', 0)
        hsz = int(kwargs['hsz'])
        weight_init = kwargs.get('weight_init', 'uniform')
        num_heads = int(kwargs.get('num_heads', 4))
        return BiLSTMLANEncoder(
            input_sz,
            hsz,
            nlayers,
            len(self.labels),
            num_heads=num_heads,
            blstm_dropout=blstm_dropout,
            mha_dropout=mha_dropout,
            unif=unif,
            weight_init=weight_init,
        )


# The Scaled Dot product attention module calculates the softmax of of layer so that it can
# calculate a weighted sum of the values based on the attention scores. Currently mead isn't
# set up to allow picking loss functions for tasks so this clashes with the CrossEntropyLoss
# used by the tagger. These are special classes that skip doing that.
class TruncatedSeqScaledDotProductAttention(SequenceSequenceAttention):
    def __init__(self, pdrop=0.1, **kwargs):
        super().__init__(pdrop=pdrop, **kwargs)

    def _attention(self, query, key, mask=None):
        # (., H, T, T) = (., H, T, D) x (., H, D, T)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return scores

    def _update(self, a, _):
        return a


# This is a partial MHA class that only returns the attention weights instead of combining it
# with the values. It also removes the unneeded value and output parameters
class TruncatedMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, scale=False):
        super().__init__()
        assert num_heads == 1
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.w_Q = Dense(d_model, d_model)
        self.w_K = Dense(d_model, d_model)
        self.attn_fn = TruncatedSeqScaledDotProductAttention(dropout)

    def forward(self, qkvm):
        query, key, value, mask = qkvm
        batchsz = query.size(0)

        # (B, H, T, D)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)

        attn = self.attn_fn((query, key, value, mask))
        attn = attn.squeeze(1)
        return attn
