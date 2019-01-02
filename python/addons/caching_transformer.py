"""Currently the cache is producing the same things (within a margin of about
   1e-6 or 1e-5). However when it is used in decoding it slowly diverges. The
   first couple outputs match but as it gets longer the outputs change.
"""


import torch
import torch.nn as nn
from baseline.model import register_decoder
from baseline.pytorch.seq2seq.decoders import TransformerDecoderWrapper


class CachingSelfMultiHeadedAttention(nn.Module):
    def __init__(self, mha):
        super(CachingSelfMultiHeadedAttention, self).__init__()
        self.d_k = mha.d_k
        self.h = mha.h
        self.w_Q = mha.w_Q
        self.w_K = mha.w_K
        self.w_V = mha.w_V
        self.w_O = mha.w_O
        self.attn_fn = mha.attn_fn
        self.attn = None
        self.dropout = mha.dropout

    def forward(self, q, mask=None, cache={}):
        batchsz = q.size(0)
        query = self.w_Q(q).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_K(q).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        if 'key_pre' in cache:
            key = torch.cat([cache['key_pre'], key], dim=2)
        cache['key_pre'] = key
        value = self.w_V(q).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        if 'value_pre' in cache:
            value = torch.cat([cache['value_pre'], value], dim=2)
        cache['value_pre'] = value

        x, self.attn = self.attn_fn(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batchsz, -1, self.h * self.d_k)
        return self.w_O(x), cache


class CachingSrcMultiHeadedAttention(nn.Module):
    def __init__(self, mha):
        super(CachingSrcMultiHeadedAttention, self).__init__()
        self.d_k = mha.d_k
        self.h = mha.h
        self.w_Q = mha.w_Q
        self.w_K = mha.w_K
        self.w_V = mha.w_V
        self.w_O = mha.w_O
        self.attn_fn = mha.attn_fn
        self.attn = None
        self.dropout = mha.dropout
        self.mha = mha

    def forward(self, q, k, v, mask=None, cache={}):
        batchsz = q.size(0)
        query = self.w_Q(q).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)

        if 'key_pre' not in cache:
            key = self.w_K(k).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
            cache['key_pre'] = key
        else:
            key = cache['key_pre']

        if 'value_pre' not in cache:
            value = self.w_V(v).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
            cache['value_pre'] = value
        else:
            value = cache['value_pre']

        x, self.attn = self.attn_fn(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batchsz, -1, self.h * self.d_k)
        return self.w_O(x), cache


class CachingTransformerDecoder(nn.Module):
    def __init__(self, td):
        super(CachingTransformerDecoder, self).__init__()
        self.d_model = td.d_model
        self.self_attn = CachingSelfMultiHeadedAttention(td.self_attn)
        self.src_attn = CachingSrcMultiHeadedAttention(td.src_attn)
        self.ffn = td.feed_forward
        self.ln1 = td.ln1
        self.ln2 = td.ln2
        self.ln3 = td.ln3
        self.dropout = td.dropout

    @staticmethod
    def build_cache():
        return {'self': {}, 'src': {}}

    def forward(self, x, memory, src_mask, tgt_mask, cache):
        x = self.ln1(x)
        x_p, cache['self'] = self.self_attn(x, tgt_mask, cache['self'])
        x = x + self.dropout(x_p)

        x = self.ln2(x)
        x_p, cache['src'] = self.src_attn(x, memory, memory, src_mask, cache['src'])
        x = x + self.dropout(x_p)

        x = self.ln3(x)
        x = x + self.dropout(self.ffn(x))
        return x, cache


class CachingTransformerDecoderStack(nn.Module):
    def __init__(self, tds):
        super(CachingTransformerDecoderStack, self).__init__()
        self.layers = nn.ModuleList([CachingTransformerDecoder(layer) for layer in tds.layers])
        self.norm = tds.norm

    def forward(self, x, memory, src_mask, tgt_mask, cache=None):
        new_cache = []
        for layer, c in zip(self.layers, cache):
            x, c = layer(x, memory, src_mask, tgt_mask, c)
            new_cache.append(c)
        return self.norm(x), new_cache

    def build_cache(self):
        return [layer.build_cache() for layer in self.layers]


@register_decoder(name='cache-transformer')
class CachingTransformerDecoderWrapper(TransformerDecoderWrapper):
    def cache_forward(self, encoder_output, dst, cache):
        embed_out_bth = self.tgt_embeddings(dst)
        embed_out_bth = self.proj_to_hsz(embed_out_bth)
        context_bth = encoder_output.output
        # dst_mask = subsequent_mask(1).type_as(embed_out_bth)
        dst_mask = None
        src_mask = encoder_output.src_mask.unsqueeze(1).unsqueeze(1)
        output, cache = self.cache_decoder(embed_out_bth, context_bth, src_mask, dst_mask, cache)
        output = self.proj_to_dsz(output)
        prob = self.output(output)
        return prob, cache

    def beam_init(self, encoder_outputs, K):
        # Tile the outputs for the beams.
        encoder_outputs = TransformerEncoderOutput(
            repeat_batch(encoder_outputs.output, K),
            repeat_batch(encoder_outputs.src_mask, K)
        )
        self.cache_decoder = CachingTransformerDecoderStack(self.transformer_decoder)
        self.cache_decoder.eval()
        return encoder_outputs, self.cache_decoder.build_cache()

    def beam_step(self, paths, extra):
        encoder_outputs, cache = extra
        B, K, T = paths.size()
        last = paths[:, :, -1].view(B * K, 1)
        probs, cache = self.cache_forward(encoder_outputs, last, cache)
        return probs, (encoder_outputs, cache)


    def beam_update(self, beams, extra):
        encoder_outputs, cache = extra
        for c in cache:
            c = c['self']
            for k, v in c.items():
                c[k] = v[beams]
        return encoder_outputs, cache
