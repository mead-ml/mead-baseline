import numpy as np
import dynet as dy
from baseline.utils import Offsets, topk
from baseline.dy.dynety import DynetModel, Linear, WeightShareLinear, squeeze_and_transpose, attention
from baseline.dy.transformer import subsequent_mask, TransformerDecoderStack
from baseline.model import (
    register_decoder,
    register_arc_policy,
    create_seq2seq_arc_policy
)


class ArcPolicy(object):
    def __init__(self):
        super(ArcPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        pass

    def __call__(self, encoder_output, hsz, beam_width=1):
        h_i = self.get_state(encoder_output)
        context = encoder_output.output
        if beam_width > 1:
            # To vectorize, we need to expand along the batch dimension, K times
            context = [dy.concatenate_to_batch([c] * beam_width) for c in context]
            h_i = [dy.concatenate_to_batch([h] * beam_width) for h in h_i]
        _, batchsz = context[0].dim()
        init_zeros = dy.zeros((hsz,), batch_size=batchsz)
        return h_i, init_zeros, context


@register_arc_policy(name='default')
class TransferLastHiddenPolicy(ArcPolicy):
    def __init__(self):
        super(TransferLastHiddenPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        return encoder_outputs.hidden


@register_arc_policy(name='no_arc')
class NoArcPolicy(ArcPolicy):
    def __init__(self):
        super(NoArcPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        final_state = encoder_outputs.hidden
        shape, batchsz = final_state[0].dim()
        return [dy.zeros(shape, batch_size=batchsz) for _ in len(final_state)]


class DecoderBase(DynetModel):
    def __init__(self, pc):
        super(DecoderBase, self).__init__(pc)

    def __call__(self, encoder_output, dst, train):
        pass

    def predict_one(self, src, encoder_outputs, **kwargs):
        pass


@register_decoder(name='vanilla')
class RNNDecoder(DecoderBase):
    def __init__(self, tgt_embeddings, **kwargs):
        pc = kwargs.pop('pc').add_subcollection(name=kwargs.get('name', 'rnn-decoder'))
        super(RNNDecoder, self).__init__(pc)
        self.hsz = kwargs['hsz']
        self.arc_policy = create_seq2seq_arc_policy(**kwargs)
        self.tgt_embeddings = tgt_embeddings
        rnntype = kwargs.get('rnntype', 'lstm')
        layers = kwargs['layers']
        feed_input = kwargs.get('feed_input', True)
        dsz = tgt_embeddings.get_dsz()
        if feed_input:
            self.input_i = self._feed_input
            dsz += self.hsz
        else:
            self.input_i = self._basic_input
        self.pdrop = kwargs.get('dropout', 0.5)
        self.decoder_rnn = dy.VanillaLSTMBuilder(layers, dsz, self.hsz, self.pc)
        self.init_attn(**kwargs)

        do_weight_tying = bool(kwargs.get('tie_weights', False))

        if do_weight_tying:
            if self.hsz == tgt_embeddings.get_dsz():
                self.preds = WeightShareLinear(tgt_embeddings.get_vsz(), tgt_embeddings.embeddings, self.pc, transform=squeeze_and_transpose, name=tgt_embeddings.pc.name())
            else:
                raise ValueError("weight tying only valid when prediction projection \
layer's hidden size == embedding weight dimensions")
        else:
            self.preds = Linear(self.tgt_embeddings.get_vsz(), self.hsz, self.pc)

    @staticmethod
    def _basic_input(dst_embed_i, _):
        return dst_embed_i

    @staticmethod
    def _feed_input(dst_embed_i, attn_output_i):
        return dy.concatenate([dst_embed_i, attn_output_i])

    def init_attn(self, **kwargs):
        pass

    def attn_cache(self, context):
        pass

    def attn(self, query, mask):
        return query

    def __call__(self, encoder_outputs, dst, train=False):
        src_mask = encoder_outputs.src_mask
        h_i, output_i, context = self.arc_policy(encoder_outputs, self.hsz)
        output = self.decode_rnn(context, h_i, output_i, dst, src_mask, train)
        return self.output(output)

    def decode_rnn(self, context, h_i, output_i, dst, src_mask, train):
        if train:
            self.decoder_rnn.set_dropout(self.pdrop)
        else:
            self.decoder_rnn.disable_dropout()
        embed_out = self.tgt_embeddings.encode(dst, train)
        outputs = []
        self.attn_cache(context)
        rnn_state = self.decoder_rnn.initial_state(h_i)
        for embed_i in embed_out:
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            output_i = self.attn(rnn_output_i, src_mask)
            outputs.append(output_i)
        return outputs

    def output(self, x):
        return [self.preds(y) for y in x]

    def prediction(self, x):
        return [dy.log_softmax(y) for y in self.output(x)]

    def predict_one(self, src, encoder_outputs, **kwargs):
        K = int(kwargs.get('beam', 5))
        mxlen = int(kwargs.get('mxlen', 100))

        paths = [[Offsets.GO] for _ in range(K)]
        # Which beams are done?
        done = np.array([False] * K)
        scores = np.array([0.0]*K)
        hidden, output_i, context = self.arc_policy(encoder_outputs, self.hsz, beam_width=K)
        num_states = len(hidden)
        rnn_state = self.decoder_rnn.initial_state(hidden)
        self.attn_cache(context)
        src_mask = encoder_outputs.src_mask

        for i in range(mxlen):
            dst_last = np.array([path[-1] for path in paths]).reshape(1, K)
            embed_i = self.tgt_embeddings.encode(dst_last)[-1]
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            output_i = self.attn(rnn_output_i, src_mask)
            wll = self.prediction([output_i])[-1].npvalue()  # (V,) K
            V = wll.shape[0]
            if i > 0:
                # expanded_history = np.expand_dims(scores, -1)
                # done_mask = np.expand_dims((done == False).astype(np.uint8), -1)
                # sll = np.multiply(wll.T, done_mask) + expanded_history

                wll = wll.T
                expanded_history = np.expand_dims(scores, -1)
                done_mask = np.expand_dims((done == False).astype(np.uint8), -1)
                done_mask_inv = (done_mask != 1).astype(np.uint8)
                eos_mask = np.zeros((1, V)).astype(np.uint8)
                mask = ((done_mask & eos_mask) != 1).astype(np.uint8)
                masked_wll = np.multiply(done_mask, wll)
                negged_wll = masked_wll + (done_mask_inv * -1e4)
                removed_eos = np.multiply(mask, negged_wll)
                sll = removed_eos + expanded_history
            else:
                sll = wll.T

            flat_sll = sll.reshape(-1)

            bests = topk(K, flat_sll)
            best_idx_flat = np.array(list(bests.keys()))
            best_beams = best_idx_flat // V
            best_idx = best_idx_flat % V

            new_paths = []
            new_done = []

            hidden = rnn_state.s()
            new_hidden = [[] for _ in range(num_states)]
            for j, best_flat in enumerate(best_idx_flat):
                beam_id = best_beams[j]
                best_word = best_idx[j]
                if done[j]:
                    new_paths.append(paths[beam_id] + [Offsets.EOS])
                else:
                    new_paths.append(paths[beam_id] + [best_word])
                if best_word == Offsets.EOS:
                    done[j] = True
                new_done.append(done[beam_id])
                scores[j] = bests[best_flat]
                # For each path, we need to pick that out and add it to the hiddens
                # This will be (c1, c2, ..., h1, h2, ...)
                for h_i, h in enumerate(hidden):
                    new_hidden[h_i].append(dy.pick_batch_elem(h, beam_id))

            done = np.array(new_done)
            new_hidden = [dy.concatenate_to_batch(new_h) for new_h in new_hidden]
            paths = new_paths
            rnn_state = self.decoder_rnn.initial_state(new_hidden)

        paths = np.stack([p[1:] for p in paths])
        return paths, scores


@register_decoder(name='default')
class RNNDecoderWithAttn(RNNDecoder):
    def __init__(self, tgt_embeddings, **kwargs):
        super(RNNDecoderWithAttn, self).__init__(tgt_embeddings, **kwargs)

    def init_attn(self, **kwargs):
        attn_type = kwargs.get('attn_type', 'bahdanau')
        self.attn_module = attention(attn_type, self.hsz, self.pc)

    def attn_cache(self, context):
        self.attn_module.cache_encoder(context)

    def attn(self, query, mask):
        return self.attn_module(query, mask)


@register_decoder(name='transformer')
class TransformerDecoderWrapper(DecoderBase):
    def __init__(self, tgt_embedding, dropout=0.5, layers=1, hsz=None, num_heads=4, scale=True, name='transformer-decoder-wrapper', **kwargs):
        pc = kwargs['pc'].add_subcollection(name=name)
        super(TransformerDecoderWrapper, self).__init__(pc)
        self.tgt_embedding = tgt_embedding
        dsz = self.tgt_embedding.get_dsz()
        hsz = dsz if hsz is None else hsz
        self.transformer_decoder = TransformerDecoderStack(num_heads, d_model=hsz, pdrop=dropout, scale=scale, layers=layers, pc=self.pc)
        self.proj_to_hsz = Linear(hsz, dsz, self.pc) if dsz != hsz else lambda x: x
        self.proj_to_dsz = WeightShareLinear(dsz, self.proj_to_hsz.weight, self.pc, transform=dy.transpose, name=self.proj_to_hsz.pc.name()) if dsz != hsz else lambda x: x
        self.preds = Linear(self.tgt_embedding.get_vsz(), dsz, self.pc)

    def output(self, x):
        return [self.preds(y) for y in dy.transpose(x)]

    def __call__(self, encoder_output, dst, train):
        embed_out_th_b = self.tgt_embedding.encode(dst)
        embed_out_ht_b = dy.transpose(embed_out_th_b)
        embed_out_ht_b = self.proj_to_hsz(embed_out_ht_b)
        context = dy.concatenate_cols(encoder_output.output)
        T = embed_out_ht_b.dim()[0][1]
        dst_mask = subsequent_mask(T)
        src_mask = encoder_output.src_mask
        output = self.transformer_decoder(embed_out_ht_b, context, src_mask, dst_mask, train)
        output = self.proj_to_dsz(output)
        return self.output(output)

    def predict_one(self, src, encoder_outputs, **kwargs):
        mxlen = int(kwargs.get('mxlen', 100))
        ys = np.full((1, 1), Offsets.GO)
        for i in range(mxlen - 1):
            probs = self(encoder_outputs, ys, False)
            next_word = np.argmax(probs[-1].npvalue())
            ys = np.concatenate([ys, np.full((1, 1), next_word)], axis=0)
            if next_word == Offsets.EOS:
                break
        return ys.transpose(), None
