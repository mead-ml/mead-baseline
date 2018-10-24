import torch
import torch.nn as nn
from torch.autograd import Variable
from baseline.pytorch.torchy import *
from baseline.pytorch.transformer import *
from baseline.model import EncoderDecoderModel, register_model
from baseline.utils import Offsets
import os


class EncoderDecoderModelBase(nn.Module, EncoderDecoderModel):

    INPUT_BT = 0
    INPUT_TB = 1

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        super(EncoderDecoderModelBase, self).__init__()
        self.input_format = EncoderDecoderModelBase.INPUT_BT
        self.beam_sz = kwargs.get('beam', 1)
        self.gpu = kwargs.get('gpu', True)
        src_dsz, tgt_dsz = self.init_embed(src_embeddings, tgt_embedding)
        self.src_lengths_key = kwargs.get('src_lengths_key')
        self.init_encoder(src_dsz, **kwargs)
        self.init_decoder(tgt_dsz, **kwargs)
        tgt_vsz = self.tgt_embedding.get_vsz()
        dec_hsz = kwargs['hsz']
        self.preds = nn.Linear(dec_hsz, tgt_vsz)
        self.probs = nn.LogSoftmax(dim=1)

    def init_embed(self, src_embeddings, tgt_embedding, **kwargs):
        """This is the hook for providing embeddings.  It takes in a dictionary of `src_embeddings` and a single
        tgt_embedding` of type `PyTorchEmbedding`

        :param src_embeddings: (``dict``) A dictionary of PyTorchEmbeddings, one per embedding
        :param tgt_embedding: (``PyTorchEmbeddings``) A single PyTorchEmbeddings object
        :param kwargs:
        :return: Return the aggregate embedding input size
        """
        self.src_embeddings = EmbeddingsContainer()
        input_sz = 0
        for k, embedding in src_embeddings.items():
            self.src_embeddings[k] = embedding
            input_sz += embedding.get_dsz()

        self.tgt_embedding = tgt_embedding
        return input_sz, self.tgt_embedding.get_dsz()

    def init_encoder(self, input_sz, **kwargs):
        pass

    def encode(self, input, lengths):
        """

        :param input:
        :param lengths:
        :return:
        """
        pass

    def init_decoder(self, input_sz, **kwargs):
        pass

    def decode(self, encoder_outputs, dst):
        pass

    def save(self, model_file):
        """Save the model out

        :param model_file: (``str``) The filename
        :return:
        """
        torch.save(self, model_file)

    def create_loss(self):
        """Create a loss function.

        :return:
        """
        return SequenceCriterion()

    @classmethod
    def load(cls, filename, **kwargs):
        """Load a model from file

        :param filename: (``str``) The filename
        :param kwargs:
        :return:
        """
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename)
        return model

    @classmethod
    def create(cls, src_embeddings, tgt_embedding, **kwargs):
        model = cls(src_embeddings, tgt_embedding, **kwargs)
        print(model)
        return model

    def make_input(self, batch_dict):
        example = dict({})

        lengths = torch.from_numpy(batch_dict[self.src_lengths_key])
        lengths, perm_idx = lengths.sort(0, descending=True)

        if self.gpu:
            lengths = lengths.cuda()
        example['src_len'] = lengths
        for key in self.src_embeddings.keys():
            tensor = torch.from_numpy(batch_dict[key])
            tensor = tensor[perm_idx]
            if self.input_format == EncoderDecoderModelBase.INPUT_TB:
                example[key] = tensor.transpose(0, 1).contiguous()
            else:
                example[key] = tensor

            if self.gpu:
                example[key] = example[key].cuda()

        if 'tgt' in batch_dict:
            tgt = torch.from_numpy(batch_dict['tgt'])
            example['dst'] = tgt[:, :-1]
            example['tgt'] = tgt[:, 1:]
            example['dst'] = example['dst'][perm_idx]
            if self.input_format == EncoderDecoderModelBase.INPUT_TB:
                example['dst'] = example['dst'].transpose(0, 1).contiguous()
            example['tgt'] = example['tgt'][perm_idx]
            if self.gpu:
                example['dst'] = example['dst'].cuda()
                example['tgt'] = example['tgt'].cuda()
        return example

    def embed(self, input):
        all_embeddings = []
        for k, embedding in self.src_embeddings.items():
            all_embeddings.append(embedding.encode(input[k]))
        return torch.cat(all_embeddings, 2)

    def forward(self, input):
        src_len = input['src_len']
        encoder_outputs = self.encode(input, src_len)
        output = self.decode(encoder_outputs, input['dst'])
        pred = self.output(output)
        # Return as B x T x H
        return pred

    def output(self, x):
        pred = self.probs(self.preds(x.view(x.size(0) * x.size(1),
                                            -1)))
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred

    # B x K x T and here T is a list
    def predict(self, batch_dict, **kwargs):
        self.eval()
        batch = []
        # Bit of a hack
        src_field = self.src_lengths_key.split('_')[0]
        B = batch_dict[src_field].shape[0]
        for b in range(B):
            example = dict({})
            for k, value in batch_dict.items():
                example[k] = value[b].reshape((1,) + value[b].shape)
            inputs = self.make_input(example)
            batch.append(self.predict_one(inputs, **kwargs)[0])
        return batch

    def predict_one(self, inputs, **kwargs):
        pass


@register_model(task='seq2seq', name='default')
class Seq2SeqModel(EncoderDecoderModelBase):

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        """This base model is extensible for attention and other uses.  It declares minimal fields allowing the
        subclass to take over most of the duties for drastically different implementations

        :param src_embeddings: (``dict``) A dictionary of PyTorchEmbeddings
        :param tgt_embedding: (``PyTorchEmbeddings``) A single PyTorchEmbeddings object
        :param kwargs:
        """
        super(Seq2SeqModel, self).__init__(src_embeddings, tgt_embedding, **kwargs)
        self.input_format = EncoderDecoderModelBase.INPUT_TB
        self.init_attn(**kwargs)

    def init_decoder(self, input_dim, **kwargs):
        """This is the hook for providing the decoder.  It provides the input size, the rest is up to the impl.

        The default implementation provides an RNN cell, followed by a linear projection, out to a softmax

        :param input_dim: The input size
        :param kwargs:
        :return: void
        """
        dec_hsz = kwargs['hsz']
        rnntype = kwargs['rnntype']
        layers = kwargs['layers']
        feed_input = kwargs.get('feed_input', True)
        if feed_input:
            self.input_i = self._feed_input
            input_dim += dec_hsz
        else:
            self.input_i = self._basic_input
        pdrop = kwargs.get('dropout', 0.5)
        self.decoder_rnn = pytorch_rnn_cell(input_dim, dec_hsz, rnntype, layers, pdrop)
        pdrop = kwargs.get('dropout', 0.5)
        self.dropout = nn.Dropout(pdrop)

    def init_encoder(self, input_dim, **kwargs):
        """This is the hook for providing the encoder.  It provides the input size, the rest is up to the impl.

        The default implementation provides a cuDNN-accelerated RNN encoder which is optionally bidirectional

        :param input_dim: The input size
        :param kwargs:
        :return: void
        """
        enc_hsz = kwargs['hsz']
        rnntype = kwargs['rnntype']
        layers = kwargs['layers']
        self.encoder_rnn = pytorch_rnn(input_dim, enc_hsz, rnntype, layers, kwargs.get('dropout', 0.5))

    def encode(self, input, src_len):
        """

        :param input: ``torch.Tensor`` oriented TxB
        :param src_len:
        :return:
        """
        embed_in_seq = self.embed(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embed_in_seq, src_len.data.tolist())
        output_tbh, hidden = self.encoder_rnn(packed)
        output_tbh, _ = torch.nn.utils.rnn.pad_packed_sequence(output_tbh)
        T = output_tbh.shape[0]
        src_mask = sequence_mask(src_len, T).type_as(src_len.data)
        return {'output': output_tbh, 'hidden': hidden, 'src_mask': src_mask}

    def decode_rnn(self, context_tbh, h_i, output_i, dst, src_mask):
        embed_out_tbh = self.tgt_embedding(dst)
        context_bth = context_tbh.transpose(0, 1)
        outputs = []

        for i, embed_i in enumerate(embed_out_tbh.split(1)):
            # Input feeding would use previous attentional output in addition to destination embeddings
            embed_i = self.input_i(embed_i, output_i)
            output_i, h_i = self.decoder_rnn(embed_i, h_i)
            output_i = self.attn(output_i, context_bth, src_mask)
            output_i = self.dropout(output_i)
            # Attentional outputs
            outputs.append(output_i)

        outputs = torch.stack(outputs)
        return outputs, h_i

    def decode(self, encoder_outputs, dst):
        context_tbh = encoder_outputs['output']
        src_mask = encoder_outputs['src_mask']
        final_encoder_state = encoder_outputs['hidden']

        if self.gpu:
            src_mask = src_mask.cuda()
        h_i, output_i = self.bridge(final_encoder_state, context_tbh)
        output, _ = self.decode_rnn(context_tbh, h_i, output_i, dst, src_mask)
        return output.transpose(0, 1).contiguous()

    def predict_one(self, inputs, **kwargs):
        K = kwargs.get('beam', 1)
        mxlen = kwargs.get('mxlen', 100)
        with torch.no_grad():
            src_len = inputs['src_len']
            src_field = self.src_lengths_key.split('_')[0]

            encoder_outputs = self.encode(inputs, src_len)
            context = encoder_outputs['output']
            h_i = encoder_outputs['hidden']
            src_mask = encoder_outputs['src_mask']
            #context, h_i = self.encode(inputs, src_len)
            #src_mask = sequence_mask(src_len)

            paths = [[Offsets.GO] for _ in range(K)]
            # K
            scores = torch.FloatTensor([0. for _ in range(K)])
            if self.gpu:
                scores = scores.cuda()
                src_mask = src_mask.cuda()
            # TBH
            context = torch.autograd.Variable(context.data.repeat(1, K, 1))
            h_i = (torch.autograd.Variable(h_i[0].data.repeat(1, K, 1)),
                   torch.autograd.Variable(h_i[1].data.repeat(1, K, 1)))
            h_i, dec_out = self.bridge(h_i, context)

            for i in range(mxlen):
                lst = [path[-1] for path in paths]
                dst = torch.LongTensor(lst).type(inputs[src_field].data.type())
                mask_eos = dst == Offsets.EOS
                mask_pad = dst == 0
                dst = dst.view(1, K)
                var = torch.autograd.Variable(dst)
                dec_out, h_i = self.decode_rnn(context, h_i, dec_out, var, src_mask)
                # 1 x K x V
                wll = self.output(dec_out).data
                # Just mask wll against end data
                V = wll.size(-1)
                dec_out = dec_out.squeeze(0)  # get rid of T=t dimension
                # K x V
                wll = wll.squeeze(0)  # get rid of T=t dimension

                if i > 0:
                    expanded_history = scores.unsqueeze(1).expand_as(wll)
                    wll.masked_fill_(mask_eos | mask_pad, 0)
                    sll = wll + expanded_history
                else:
                    sll = wll[0]

                flat_sll = sll.view(-1)
                best, best_idx = flat_sll.squeeze().topk(K, 0)
                best_beams = best_idx / V
                best_idx = best_idx % V
                new_paths = []
                for j, beam_id in enumerate(best_beams):
                    new_paths.append(paths[beam_id] + [best_idx[j]])
                    scores[j] = best[j]

                # Copy the beam state of the winners
                for hc in h_i:  # iterate over h, c
                    old_beam_state = hc.clone()
                    for i, beam_id in enumerate(best_beams):
                        H = hc.size(2)
                        src_beam = old_beam_state.view(-1, K, H)[:, beam_id]
                        dst_beam = hc.view(-1, K, H)[:, i]
                        dst_beam.data.copy_(src_beam.data)
                paths = new_paths

            return [p[1:] for p in paths], scores

    def bridge(self, final_encoder_state, context):
        return final_encoder_state, None

    def attn(self, output_t, context, src_mask=None):
        return output_t

    def init_attn(self, **kwargs):
        pass

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


@register_model(task='seq2seq', name='attn')
class Seq2SeqAttnModel(Seq2SeqModel):

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        self.hsz = kwargs['hsz']
        super(Seq2SeqAttnModel, self).__init__(src_embeddings, tgt_embedding, **kwargs)

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

    def bridge(self, final_encoder_state, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.hsz)
        context_zeros = Variable(context.data.new(*h_size).zero_(), requires_grad=False)
        if type(final_encoder_state) is tuple:
            s1, s2 = final_encoder_state
            return (s1, s2), context_zeros
        else:
            return final_encoder_state, context_zeros


@register_model(task='seq2seq', name='transformer')
class TransformerModel(EncoderDecoderModelBase):

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        super(TransformerModel, self).__init__(src_embeddings, tgt_embedding, **kwargs)

    def encode(self, inputs, lengths):
        bth = self.embed(inputs)
        T = bth.shape[1]
        src_mask = sequence_mask(lengths, T).type_as(lengths.data).unsqueeze(1).unsqueeze(1)
        return {'output': self.transformer_encoder(bth, src_mask), 'src_mask': src_mask }

    def init_decoder(self, input_sz, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.5))
        layers = kwargs.get('layers', 1)
        d_model = int(kwargs.get('d_model', kwargs.get('hsz')))
        num_heads = kwargs.get('num_heads', 4)
        self.transformer_decoder = TransformerDecoderStack(num_heads, d_model=d_model, pdrop=pdrop, scale=True, layers=layers)

    def decode(self, encoder_output, dst):
        embed_out_bth = self.tgt_embedding(dst)
        context_bth = encoder_output['output']
        T = dst.shape[1]
        dst_mask = subsequent_mask(T).type_as(embed_out_bth)
        src_mask = encoder_output['src_mask']
        output = self.transformer_decoder(embed_out_bth, context_bth, src_mask, dst_mask)
        return output

    def init_encoder(self, input_sz, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.5))
        layers = kwargs.get('layers', 1)
        num_heads = kwargs.get('num_heads', 4)
        d_model = int(kwargs.get('d_model', kwargs.get('hsz')))
        self.transformer_encoder = TransformerEncoderStack(num_heads, d_model=d_model, pdrop=pdrop, scale=True, layers=layers)

    def predict_one(self, inputs, **kwargs):
        mxlen = kwargs.get('mxlen', 100)
        with torch.no_grad():
            src_len = inputs['src_len']
            src_field = self.src_lengths_key.split('_')[0]
            src = inputs[src_field]
            encoder_outputs = self.encode(inputs, src_len)

            # A single y value of <GO> to start
            ys = torch.ones(1, 1).fill_(Offsets.GO).type_as(src.data)

            for i in range(mxlen-1):
                # Make a mask of length T
                out = self.decode(encoder_outputs, ys)[:, -1]
                prob = self.output(out.view(1, 1, -1)).view(1, -1)
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
                # Add the word on to the end
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                if next_word == Offsets.EOS:
                    break
        return ys, None
