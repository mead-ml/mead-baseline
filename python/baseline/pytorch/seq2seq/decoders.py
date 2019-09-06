import math
from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from baseline.utils import Offsets, export
from baseline.pytorch.transformer import subsequent_mask, TransformerDecoderStack
from baseline.model import register_arc_policy, register_decoder, create_seq2seq_arc_policy
from baseline.pytorch.seq2seq.encoders import TransformerEncoderOutput
from baseline.pytorch.torchy import (
    tie_weight,
    pytorch_linear,
    pytorch_rnn_cell,
    LuongDotProductAttention,
    BahdanauAttention,
    ScaledDotProductAttention,
    LuongGeneralAttention,
)

try:
    # Pytorch introduced a boolean tensor in 1.2 and now all of our masks (created via <)
    # are torch.bool, This caused problems because the eos_mask we use a uint8. Here we
    # try to get the bool type, if pytorch 1.2 is installed this works otherwise we get
    # the uint8 type. This type is used for the creation of the eos_mask
    MASK_TYPE = torch.bool
except AttributeError:
    MASK_TYPE = torch.uint8

__all__ = []
exporter = export(__all__)


class ArcPolicy(torch.nn.Module):

    def __init__(self):
        super(ArcPolicy, self).__init__()

    def forward(self, encoder_outputs, hsz, beam_width=1):
        pass


class AbstractArcPolicy(ArcPolicy):

    def __init__(self):
        super(AbstractArcPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        pass

    def forward(self, encoder_output, hsz, beam_width=1):
        h_i = self.get_state(encoder_output)
        context = encoder_output.output
        if beam_width > 1:
            with torch.no_grad():
                context = repeat_batch(context, beam_width)
                if type(h_i) is tuple:
                    h_i = repeat_batch(h_i[0], beam_width, dim=1), repeat_batch(h_i[1], beam_width, dim=1)
                else:
                    h_i = repeat_batch(h_i, beam_width, dim=1)
        batch_size = context.shape[0]
        h_size = (batch_size, hsz)
        with torch.no_grad():
            init_zeros = context.data.new(*h_size).zero_()
        return h_i, init_zeros, context


@register_arc_policy(name='default')
class TransferLastHiddenPolicy(AbstractArcPolicy):

    def __init__(self):
        super(TransferLastHiddenPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        return encoder_outputs.hidden


@register_arc_policy(name='no_arc')
class NoArcPolicy(AbstractArcPolicy):

    def __init__(self):
        super(NoArcPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        final_encoder_state = encoder_outputs.hidden
        if isinstance(final_encoder_state, tuple):
            s1, s2 = final_encoder_state
            return s1 * 0, s2 * 0
        return final_encoder_state * 0


@register_decoder(name='vanilla')
class RNNDecoder(torch.nn.Module):

    def __init__(self, tgt_embeddings, **kwargs):
        """Construct an RNN decoder.  It provides the input size, the rest is up to the impl.

        The default implementation provides an RNN cell, followed by a linear projection, out to a softmax

        :param input_dim: The input size
        :param kwargs:
        :return: void
        """
        super(RNNDecoder, self).__init__()
        self.hsz = kwargs['hsz']
        self.arc_policy = create_seq2seq_arc_policy(**kwargs)
        self.tgt_embeddings = tgt_embeddings
        rnntype = kwargs['rnntype']
        layers = kwargs['layers']
        feed_input = kwargs.get('feed_input', True)
        dsz = tgt_embeddings.get_dsz()
        if feed_input:
            self.input_i = self._feed_input
            dsz += self.hsz
        else:
            self.input_i = self._basic_input
        pdrop = kwargs.get('dropout', 0.5)
        self.decoder_rnn = pytorch_rnn_cell(dsz, self.hsz, rnntype, layers, pdrop)
        self.dropout = torch.nn.Dropout(pdrop)
        self.init_attn(**kwargs)

        do_weight_tying = bool(kwargs.get('tie_weights', False))
        is_valid_tying = self.hsz == self.tgt_embeddings.get_dsz()

        self.preds = pytorch_linear(self.hsz, self.tgt_embeddings.get_vsz())
        if do_weight_tying:
            if is_valid_tying:
                tie_weight(self.preds, self.tgt_embeddings.embeddings)
            else:
                raise ValueError("weight tying only valid when prediction projection \
layer's hidden size == embedding weight dimensions")

    @staticmethod
    def _basic_input(dst_embed_i, _):
        """
        In this function the destination embedding is passed directly to into the decoder.  The output of previous H
        is ignored.  This is implemented using a bound method to a field in the class for speed so that this decision
        is handled at initialization, not as a conditional in the training or inference

        :param embed_i: The embedding at i
        :param _: Ignored
        :return: basic input
        """
        return dst_embed_i.squeeze(0)

    @staticmethod
    def _feed_input(embed_i, attn_output_i):
        """
        In this function the destination embedding is concatenated with the previous attentional output and
        passed to the decoder. This is implemented using a bound method to a field in the class for speed
        so that this decision is handled at initialization, not as a conditional in the training or inference

        :param embed_i: The embedding at i
        :param output_i: This is the last H state
        :return: an input that is a concatenation of previous state and destination embedding
        """
        embed_i = embed_i.squeeze(0)
        return torch.cat([embed_i, attn_output_i], 1)

    def forward(self, encoder_outputs, dst):
        src_mask = encoder_outputs.src_mask
        h_i, output_i, context_bth = self.arc_policy(encoder_outputs, self.hsz)
        output_tbh, _ = self.decode_rnn(context_bth, h_i, output_i, dst.transpose(0, 1), src_mask)
        pred = self.output(output_tbh)
        return pred.transpose(0, 1).contiguous()

    def decode_rnn(self, context_bth, h_i, output_i, dst_tbh, src_mask):
        embed_out_tbh = self.tgt_embeddings(dst_tbh)

        outputs = []

        for i, embed_i in enumerate(embed_out_tbh.split(1)):
            # Input feeding would use previous attentional output in addition to destination embeddings
            embed_i = self.input_i(embed_i, output_i)
            output_i, h_i = self.decoder_rnn(embed_i, h_i)
            output_i = self.attn(output_i, context_bth, src_mask)
            output_i = self.dropout(output_i)
            # Attentional outputs
            outputs.append(output_i)

        outputs_tbh = torch.stack(outputs)
        return outputs_tbh, h_i

    def attn(self, output_t, context, src_mask=None):
        return output_t

    def init_attn(self, **kwargs):
        pass

    def output(self, x):
        pred = F.log_softmax(self.preds(x.view(x.size(0)*x.size(1), -1)), dim=-1)
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred

    def beam_init(self, encoder_outputs, K):
        """Tile batches for encoder inputs and the likes."""
        src_mask = repeat_batch(encoder_outputs.src_mask, K)
        h_i, dec_out, context = self.arc_policy(encoder_outputs, self.hsz, K)
        return h_i, dec_out, context, src_mask

    def beam_step(self, paths, extra):
        """Calculate the probs of the next output and update state."""
        h_i, dec_out, context, src_mask = extra
        last = paths[:, :, -1].view(1, -1)
        dec_out, h_i = self.decode_rnn(context, h_i, dec_out, last, src_mask)
        probs = self.output(dec_out)
        dec_out = dec_out.squeeze(0)
        return probs, (h_i, dec_out, context, src_mask)

    def beam_update(self, beams, extra):
        """Select the correct hidden states and outputs to used based on the best performing beams."""
        h_i, dec_out, context, src_mask = extra
        h_i = tuple(hc[:, beams, :] for hc in h_i)
        dec_out = dec_out[beams, :]
        return h_i, dec_out, context, src_mask

    def beam_search(self, encoder_outputs, **kwargs):
        alpha = kwargs.get('alpha')
        if alpha is not None: kwargs['length_penalty'] = partial(gnmt_length_penalty, alpha=alpha)
        return beam_search(encoder_outputs, self.beam_init, self.beam_step, self.beam_update, **kwargs)


@register_decoder(name='default')
class RNNDecoderWithAttn(RNNDecoder):

    def __init__(self, tgt_embeddings, **kwargs):
        super(RNNDecoderWithAttn, self).__init__(tgt_embeddings, **kwargs)

    def init_attn(self, **kwargs):
        attn_type = kwargs.get('attn_type', 'bahdanau').lower()
        if attn_type == 'dot':
            self.attn_module = LuongDotProductAttention(self.hsz)
        elif attn_type == 'concat' or attn_type == 'bahdanau':
            self.attn_module = BahdanauAttention(self.hsz)
        elif attn_type == 'sdp':
            self.attn_module = ScaledDotProductAttention(self.hsz)
        else:
            self.attn_module = LuongGeneralAttention(self.hsz)

    def attn(self, output_t, context, src_mask=None):
        return self.attn_module(output_t, context, context, src_mask)


@register_decoder(name='transformer')
class TransformerDecoderWrapper(torch.nn.Module):

    def __init__(self, tgt_embeddings, dropout=0.5, layers=1, hsz=None, num_heads=4, scale=True, **kwargs):
        super(TransformerDecoderWrapper, self).__init__()
        self.tgt_embeddings = tgt_embeddings
        dsz = self.tgt_embeddings.get_dsz()
        if hsz is None:
            hsz = dsz

        self.transformer_decoder = TransformerDecoderStack(num_heads, d_model=hsz, pdrop=dropout, scale=scale, layers=layers)

        self.proj_to_dsz = self._identity
        self.proj_to_hsz = self._identity
        if hsz != dsz:
            self.proj_to_hsz = pytorch_linear(dsz, hsz)
            self.proj_to_dsz = pytorch_linear(hsz, dsz)
            del self.proj_to_dsz.weight
            self.proj_to_dsz.weight = torch.nn.Parameter(self.proj_to_hsz.weight.transpose(0, 1), requires_grad=True)

        self.preds = pytorch_linear(dsz, self.tgt_embeddings.get_vsz())

        do_weight_tying = bool(kwargs.get('tie_weights', False))
        if do_weight_tying:
            self.preds.weight = self.tgt_embeddings.weight.transpose(0, 1)


    def _identity(self, x):
        return x

    def forward(self, encoder_output, dst):
        embed_out_bth = self.tgt_embeddings(dst)
        embed_out_bth = self.proj_to_hsz(embed_out_bth)
        context_bth = encoder_output.output
        T = embed_out_bth.shape[1]
        dst_mask = subsequent_mask(T).type_as(embed_out_bth)
        src_mask = encoder_output.src_mask.unsqueeze(1).unsqueeze(1)
        output = self.transformer_decoder(embed_out_bth, context_bth, src_mask, dst_mask)
        output = self.proj_to_dsz(output)
        prob = self.output(output)
        return prob

    def output(self, x):
        pred = F.log_softmax(self.preds(x.view(x.size(0)*x.size(1), -1)), dim=-1)
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred

    def beam_init(self, encoder_outputs, K):
        """Tile for the batch of the encoder inputs."""
        encoder_outputs = TransformerEncoderOutput(
            repeat_batch(encoder_outputs.output, K),
            repeat_batch(encoder_outputs.src_mask, K)
        )
        return encoder_outputs

    def beam_step(self, paths, extra):
        """Calculate the probs for the last item based on the full path."""
        B, K, T = paths.size()
        return self(extra, paths.view(B * K, T))[:, -1], extra

    def beam_update(self, beams, extra):
        """There is no state for the transformer so just pass it."""
        return extra

    def beam_search(self, encoder_outputs, **kwargs):
        return beam_search(encoder_outputs, self.beam_init, self.beam_step, self.beam_update, **kwargs)


def update_lengths(lengths, eoses, idx):
    """Update the length of a generated tensor based on the first EOS found.

    This is useful for a decoding situation where tokens after an EOS
    can be something other than EOS. This also makes sure that a second
    generated EOS doesn't effect the lengths.

    :param lengths: `torch.LongTensor`: The lengths where zero means an
        unfinished sequence.
    :param eoses:  `torch.ByteTensor`: A mask that has 1 for sequences that
        generated an EOS.
    :param idx: `int`: What value to fill the finished lengths with (normally
        the current decoding timestep).

    :returns: `torch.Tensor`: The updated lengths tensor (same shape and type).
    """
    # If a length is 0 it has never had a length set so it is eligible to have
    # this EOS be the length.
    updatable_lengths = (lengths == 0)
    # If this length can be updated AND this token is an eos
    lengths_mask = updatable_lengths & eoses
    return lengths.masked_fill(lengths_mask, idx)


def gnmt_length_penalty(lengths, alpha=0.8):
    """Calculate a length penalty from https://arxiv.org/pdf/1609.08144.pdf

    The paper states the penalty as (5 + |Y|)^a / (5 + 1)^a. This is impelmented
    as ((5 + |Y|) / 6)^a for a (very) tiny performance boost

    :param lengths: `torch.LongTensor`: [B, K] The lengths of the beams.
    :param alpha: `float`: A hyperparameter. See Table 2 for a search on this
        parameter.

    :returns:
        `torch.FloatTensor`: [B, K, 1] The penalties.
    """
    lengths = lengths.to(torch.float)
    penalty = torch.pow(((5 + lengths) / 6), alpha)
    return penalty.unsqueeze(-1)


def no_length_penalty(lengths):
    """A dummy function that returns a no penalty (1)."""
    return torch.ones_like(lengths).to(torch.float).unsqueeze(-1)


def repeat_batch(t, K, dim=0):
    """Repeat a tensor while keeping the concept of a batch.

    :param t: `torch.Tensor`: The tensor to repeat.
    :param K: `int`: The number of times to repeat the tensor.
    :param dim: `int`: The dimension to repeat in. This should be the
        batch dimension.

    :returns: `torch.Tensor`: The repeated tensor. The new shape will be
        batch size * K at dim, the rest of the shapes will be the same.

    Example::

        >>> a = torch.arange(10).view(2, -1)
        >>> a
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> a.repeat(2, 1)
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> repeat_batch(a, 2)
	tensor([[0, 1, 2, 3, 4],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[5, 6, 7, 8, 9]])
    """
    shape = t.shape
    tiling = [1] * (len(shape) + 1)
    tiling[dim + 1] = K
    tiled = t.unsqueeze(dim + 1).repeat(tiling)
    old_bsz = shape[dim]
    new_bsz = old_bsz * K
    new_shape = list(shape[:dim]) + [new_bsz] + list(shape[dim+1:])
    return tiled.view(new_shape)


def beam_search(
        encoder_outputs,
        init, step, update,
        length_penalty=no_length_penalty,
        **kwargs
):
    """Perform batched Beam Search.

    Note:
        The paths and lengths generated do not include the <GO> token.

    :param encoder_outputs: `namedtuple` The outputs of the encoder class.
    :param init: `Callable(ecnoder_outputs: encoder_outputs, K: int)` -> Any: A
        callable that is called once at the start of the search to initialize
        things. This returns a blob that is passed to other callables.
    :param step: `Callable(paths: torch.LongTensor, extra) -> (probs: torch.FloatTensor, extra):
        A callable that is does a single decoding step. It returns the log
        probabilities over the vocabulary in the last dimension. It also returns
        any state the decoding process needs.
    :param update: `Callable(beams: torch.LongTensor, extra) -> extra:
        A callable that is called to edit the decoding state based on the selected
        best beams.
    :param length_penalty: `Callable(lengths: torch.LongTensor) -> torch.floatTensor
        A callable that generates a penalty based on the lengths. Lengths is
        [B, K] and the returned penalty should be [B, K, 1] (or [B, K, V] to
        have token based penalties?)

    :Keyword Arguments:
    * *beam* -- `int`: The number of beams to use.
    * *mxlen* -- `int`: The max number of steps to run the search for.

    :returns:
        tuple(preds: torch.LongTensor, lengths: torch.LongTensor, scores: torch.FloatTensor)
        preds: The predicted values: [B, K, max(lengths)]
        lengths: The length of each prediction [B, K]
        scores: The score of each path [B, K]
    """
    K = kwargs.get('beam', 5)
    mxlen = kwargs.get('mxlen', 100)
    bsz = encoder_outputs.output.size(0)
    device = encoder_outputs.output.device
    with torch.no_grad():
        extra = init(encoder_outputs, K)
        paths = torch.full((bsz, K, 1), Offsets.GO, dtype=torch.long, device=device)
        # This tracks the log prob of each beam. This is distinct from score which
        # is based on the log prob and penalties.
        log_probs = torch.zeros((bsz, K), dtype=torch.float, device=device)
        # Tracks the lengths of the beams, unfinished beams have a lengths of zero.
        lengths = torch.zeros((bsz, K), dtype=torch.long, device=device)
        last = paths[:, :, -1]  # [B, K]

        for i in range(mxlen - 1):
            probs, extra = step(paths, extra)
            V = probs.size(-1)
            probs = probs.view((bsz, K, V))  # [B, K, V]
            if i > 0:
                # This mask is for all beams that are done.
                done_mask = (lengths != 0).unsqueeze(-1)  # [B, K, 1]
                # Can creating this mask be moved out of the loop? It never changes but we don't have V
                # This mask selects the EOS token
                eos_mask = torch.zeros((1, 1, V), dtype=MASK_TYPE, device=device)
                eos_mask[:, :, Offsets.EOS] = 1
                # This mask selects the EOS token of only the beams that are done.
                mask = done_mask & eos_mask
                # Put all probability mass on the EOS token for finished beams.
                # Otherwise as the other beams get longer they will all give
                # up and eventually select this beam and all outputs become
                # the same.
                probs = probs.masked_fill(done_mask, -np.inf)
                probs = probs.masked_fill(mask, 0)
                probs = log_probs.unsqueeze(-1) + probs  # [B, K, V]
                # Calculate the score of the beam based on the current length.
                path_scores = probs / length_penalty(lengths.masked_fill(lengths==0, i+1))
            else:
                # On the first step we only look at probabilities for the first beam.
                # If we don't then the probs will be the same for each beam
                # This means the same token will be selected for each beam
                # And we won't get any diversity.
                # Using only the first beam ensures K different starting points.
                path_scores = probs[:, 0, :]

            flat_scores = path_scores.view(bsz, -1)  # [B, K * V]
            best_scores, best_idx = flat_scores.topk(K, 1)
            # Get the log_probs of the best scoring beams
            log_probs = probs.view(bsz, -1).gather(1, best_idx).view(bsz, K)

            best_beams = best_idx / V  # Get which beam it came from
            best_idx = best_idx % V  # Get the index of the word regardless of which beam it is.

            # Best Beam index is relative within the batch (only [0, K)).
            # This makes the index global (e.g. best beams for the second
            # batch example is in [K, 2*K)).
            offsets = torch.arange(bsz, dtype=torch.long, device=device) * K
            offset_beams = best_beams + offsets.unsqueeze(-1)
            flat_beams = offset_beams.view(bsz * K)
            # Select the paths to extend based on the best beams
            flat_paths = paths.view(bsz * K, -1)
            new_paths = flat_paths[flat_beams, :].view(bsz, K, -1)
            # Add the selected outputs to the paths
            paths = torch.cat([new_paths, best_idx.unsqueeze(-1)], dim=2)

            # Select the lengths to keep tracking based on the valid beams left.
            lengths = lengths.view(-1)[flat_beams].view((bsz, K))

            extra = update(flat_beams, extra)

            # Updated lengths based on if we hit EOS
            last = paths[:, :, -1]
            eoses = (last == Offsets.EOS)
            lengths = update_lengths(lengths, eoses, i + 1)
            if (lengths != 0).all():
                break
        else:
            # This runs if the loop didn't break meaning one beam hit the max len
            # Add an EOS to anything that hasn't hit the end. This makes the scores real.
            probs, extra = step(paths, extra)

            V = probs.size(-1)
            probs = probs.view((bsz, K, V))
            probs = probs[:, :, Offsets.EOS]  # Select the score of EOS
            # If any of the beams are done mask out the score of this EOS (they already had an EOS)
            probs = probs.masked_fill((lengths != 0), 0)
            log_probs = log_probs + probs
            end_tokens = torch.full((bsz, K, 1), Offsets.EOS, device=device, dtype=paths.dtype)
            paths = torch.cat([paths, end_tokens], dim=2)
            lengths = update_lengths(lengths, torch.ones_like(lengths) == 1, mxlen)
            best_scores = log_probs / length_penalty(lengths).squeeze(-1)

    # Slice off the Offsets.GO token
    paths = paths[:, :, 1:]
    return paths, lengths, best_scores
