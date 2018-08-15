import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from baseline.pytorch.torchy import *
from baseline.model import EncoderDecoder


def subsequent_mask(size):
    """Form a triangular mask to prevent looking ahead

    :param size: Size of the attention mask
    :return:
    """
    attn_shape = (1, size, size)
    subsequent_mask_ = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask_) == 0


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

    We apply the query to the keys to recieve our weights via softmax, which are then applied
    for each value, but in a series of efficient matrix operations.  In the case of self-attention,
    the key, query and values are all low order projections of the same input.

    The implementation here is from: http://nlp.seas.harvard.edu/2018/04/03/attention.html

    :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
    :param key: a set of keys from encoder or self
    :param value: a set of values from encoder or self
    :param mask: masking (for destination) to prevent seeing what we shouldnt
    :param dropout: apply dropout operator post-attention (this is not a float)
    :return: A tensor that is (BxHxTxT)

    """
    # (., H, T, T) = (., H, T, D) x (., H, D, T)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Multi-headed attention provides multiple looks of low-order projections K, Q and V using an attention function
    (specifically `scaled_dot_product_attention` in the paper.  This allows multiple relationships to be illuminated
    via attention on different positional and representational information from each head.

    The number of heads `h` times the low-order projection dim `d_k` is equal to `d_model` (which is asserted upfront).
    This means that each weight matrix can be simply represented as a linear transformation from `d_model` to `d_model`,
    and partitioned into heads after the fact.

    Finally, an output projection is applied which brings the output space back to `d_model`, in preparation for the
    sub-sequent `FFN` sub-layer.

    There are 3 uses of multi-head attention in the Transformer.
    For encoder-decoder layers, the queries come from the previous decoder layer, and the memory keys come from
    the encoder.  For encoder layers, the K, Q and V all come from the output of the previous layer of the encoder.
    And for self-attention in the decoder, K, Q and V all come from the decoder, but here it is masked to prevent using
    future values
    """
    def __init__(self, h, d_model, dropout=0.1, attn_fn=scaled_dot_product_attention):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param attn_fn: A function to apply attention, defaults to SDP
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_Q = pytorch_linear(d_model, d_model)
        self.w_K = pytorch_linear(d_model, d_model)
        self.w_V = pytorch_linear(d_model, d_model)
        self.w_O = pytorch_linear(d_model, d_model)
        self.attn_fn = attn_fn
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batchsz = query.size(0)

        # (B, H, T, D)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_V(value).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = self.attn_fn(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(batchsz, -1, self.h * self.d_k)
        return self.w_O(x)


class FFN(nn.Module):
    """
    FFN from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    The `FFN` layer is block in the Transformer that follows multi-headed self-attention.  It consists
    of an expansion from `d_model` to `d_ff` (with sub-sequent relu and dropout), followed by a squeeze
    layer that pushes it back to `d_model`.  In the `tensor2tensor` codebase, this is implemented as convolution of
    size 1 over the temporal sequence, which is equivalent, but in PyTorch, we dont need to do anything explicitly,
    thanks to https://github.com/pytorch/pytorch/pull/1935!

    """
    def __init__(self, d_model, d_ff, pdrop=0.1):
        """Constructor, takes in model size (which is the external currency of each block) and the feed-forward size

        :param d_model: The model size.  This is the size passed through each block
        :param d_ff: The feed-forward internal size, which is typical 4x larger, used internally
        :param pdrop: The probability of dropping output
        """
        super(FFN, self).__init__()
        self.expansion = pytorch_linear(d_model, d_ff)
        self.squeeze = pytorch_linear(d_ff, d_model)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x):
        """Expand to `d_ff` then activation, followed by a squeeze operation back down to `d_model`

        :param x: The output of the previous attention module
        :return: An output the same size as the input, but transformed
        """
        return self.squeeze(self.dropout(torch.nn.functional.relu(self.expansion(x))))


class PosEmbeddings(nn.Module):
    """
    This implementation is from http://nlp.seas.harvard.edu/2018/04/03/attention.html, and follows the paper closely.

    Since we are not doing any temporal operations, as we do in an RNN, we have no structural knowledge of time
    from the model itself. Positional embeddings allow us to track temporal aspect of sequence by adding a bunch of
    sines and cosines to the signal.

    After staring at T2T code for quite a while, I feel this version more obviously follows the paper than the
    version that lives in T2T.  I have modified it to include the "normalization" step on the input embeddings
    internally as well.
    """
    def __init__(self, embed, d_model, dropout, mxlen, max_timescale=1.0e4, gpu=False):
        """

        :param embed: The input is a normal embedding object
        :param d_model: The size of the model (hidden units).  Currency for the Transformer blocks
        :param dropout (``float``): The amount of dropout to apply
        :param mxlen: The maximum length of the temporal sequence
        :param max_timescale:  Paper-specifies as 1.0e4, T2T allows other options
        """
        super(PosEmbeddings, self).__init__()
        self.embed = pytorch_embedding(embed)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        log_timescale_increment = math.log(max_timescale) / d_model
        inv_timescales = torch.exp(torch.arange(0, d_model, 2) * -log_timescale_increment)

        pe = torch.zeros(mxlen, d_model)
        position = torch.arange(0, mxlen).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * inv_timescales)
        pe[:, 1::2] = torch.cos(position * inv_timescales)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.gpu = gpu

    def cuda(self, device=None):
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        if self.gpu:
            return self._apply(lambda t: t.cuda(device))
        return self

    def forward(self, x):
        """Add a positional encoding to the embedding, followed by dropout

        :param x: The temporal signal in, to which the positional embeddings are applied
        :return: Embedded output
        """
        x = self.embed(x) * math.sqrt(self.d_model)
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, pdrop):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadedAttention(num_heads, d_model)
        self.ffn = FFN(d_model, d_ff, pdrop)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x, mask):
        x = self.ln1(x)
        x = x + self.dropout(self.self_attn(x, x, x, mask))

        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, pdrop):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadedAttention(num_heads, d_model)
        self.src_attn = MultiHeadedAttention(num_heads, d_model)
        self.feed_forward = FFN(d_model, d_ff, pdrop)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x, memory, src_mask, tgt_mask):

        x = self.ln1(x)
        x = x + self.dropout(self.self_attn(x, x, x, tgt_mask))

        x = self.ln2(x)
        x = x + self.dropout(self.src_attn(x, memory, memory, src_mask))

        x = self.ln3(x)
        x = x + self.dropout(self.feed_forward(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = pytorch_clone_module(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = pytorch_clone_module(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module, EncoderDecoder):

    def __init__(self, embeddings_in, embeddings_out, **kwargs):
        super(Transformer, self).__init__()
        self.gpu = kwargs.get('gpu', True)
        self.nc = embeddings_out.vsz + 1
        self.vocab1 = embeddings_in.vocab
        self.vocab2 = embeddings_out.vocab
        self.beam_sz = 1
        self.num_heads = kwargs.get('num_heads', 8)
        self.d_model = kwargs.get('d_model', 512)
        self.d_ff = kwargs.get('d_ff', 2048)
        self.nlayers = kwargs.get('layers', kwargs.get('nlayers', 6))
        self.dropout = kwargs.get('dropout', 0.5)
        emb_on_gpu = kwargs.get('embeddings_gpu', False)
        mxlen = kwargs.get('mxlen', 100)
        self.pos_in = PosEmbeddings(embeddings_in, self.d_model, self.dropout, mxlen, gpu=emb_on_gpu)
        self.pos_out = PosEmbeddings(embeddings_out, self.d_model, self.dropout, mxlen, gpu=emb_on_gpu)
        self.encoder = Encoder(EncoderLayer(self.num_heads, self.d_model, self.d_ff, self.dropout), self.nlayers)
        self.decoder = Decoder(DecoderLayer(self.num_heads, self.d_model, self.d_ff, self.dropout), self.nlayers)
        self.preds = nn.Linear(self.d_model, self.nc)

    def get_src_vocab(self):
        return self.vocab1

    def get_dst_vocab(self):
        return self.vocab2

    def save(self, model_file):
        torch.save(self, model_file)

    def create_loss(self):
        return SequenceCriterion()

    @classmethod
    def load(cls, outname, **kwargs):
        model = torch.load(outname)
        return model

    @classmethod
    def create(cls, input_embeddings, output_embeddings, **kwargs):

        model = cls(input_embeddings, output_embeddings, **kwargs)
        return model

    def make_input(self, batch_dict):
        src = batch_dict['src']
        tgt = batch_dict['dst']

        if tgt is not None:
            dst = tgt[:, :-1]
            tgt = tgt[:, 1:]

        if self.gpu:
            src = src.cuda()
            dst = dst.cuda()
            tgt = tgt.cuda()

        return Variable(src), Variable(dst), Variable(tgt)

    # Input better be xch, x
    def forward(self, input):
        src = input[0]
        dst = input[1]
        PAD = 0
        src_mask = (src != PAD).unsqueeze(-2)
        dst_mask = (dst != PAD).unsqueeze(-2)
        dst_mask = dst_mask & Variable(subsequent_mask(dst.size(-1)).type_as(dst_mask.data))
        encoded = self.encode(src, src_mask)
        decoded = self.decode(encoded, src_mask, dst, dst_mask)
        return self.prediction(decoded)

    def encode(self, src, src_mask):
        pos_embed = self.pos_in(src)
        encoded = self.encoder(pos_embed, src_mask)
        return encoded

    def decode(self, encoded, src_mask, dst, dst_mask):
        pos_embed = self.pos_out(dst)
        return self.decoder(pos_embed, encoded, src_mask, dst_mask)

    def prediction(self, output):
        # Reform batch as (T x B, D)
        pred = F.log_softmax(self.preds(output.view(output.size(0)*output.size(1),
                                                 -1)))
        pred = pred.view(output.size(0), output.size(1), -1)
        return pred

    # TODO, replace with a beam decoder
    def greedy_decode(self, src, mxlen):
        GO = self.vocab2['<GO>']
        EOS = self.vocab2['<EOS>']
        PAD = 0
        src_mask = (src != PAD).unsqueeze(-2)

        memory = self.encode(src, src_mask)
        # A single y value of <GO> to start
        ys = torch.ones(1, 1).fill_(GO).type_as(src.data)

        for i in range(mxlen-1):
            # Make a mask of length T
            out = self.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))[:, -1]
            prob = self.prediction(out.view(1, 1, -1)).view(1, -1)
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            # Add the word on to the end
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == EOS:
                break
        return ys

    def run(self, batch_dict, **kwargs):
        src = batch_dict['src']
        src = torch.from_numpy(src) if type(src) == np.ndarray else src
        if torch.is_tensor(src):
            src = torch.autograd.Variable(src, requires_grad=False)

        if self.gpu:
            src = src.cuda()

        batch = []
        for src_i in src:
            batch += [self.greedy_decode(src_i.view(1, -1), kwargs.get('mxlen', 100))[:, 1:]]

        return batch


def create_model(src_vocab_embed, dst_vocab_embed, **kwargs):
    model = Transformer.create(src_vocab_embed, dst_vocab_embed, **kwargs)
    return model


def load_model(modelname, **kwargs):
    return Transformer.load(modelname, **kwargs)