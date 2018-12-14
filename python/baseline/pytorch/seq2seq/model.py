from baseline.pytorch.torchy import *
from baseline.pytorch.transformer import *
from baseline.model import EncoderDecoderModel, register_model, create_seq2seq_encoder, create_seq2seq_decoder
from baseline.pytorch.seq2seq.encoders import *
from baseline.pytorch.seq2seq.decoders import *

import os


class EncoderDecoderModelBase(nn.Module, EncoderDecoderModel):

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        super(EncoderDecoderModelBase, self).__init__()
        self.beam_sz = kwargs.get('beam', 1)
        self.gpu = kwargs.get('gpu', True)
        src_dsz = self.init_embed(src_embeddings, tgt_embedding)
        self.src_lengths_key = kwargs.get('src_lengths_key')
        self.dropin_values = kwargs.get('dropin', {})
        self.encoder = self.init_encoder(src_dsz, **kwargs)
        self.decoder = self.init_decoder(tgt_embedding, **kwargs)

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
        return input_sz

    def init_encoder(self, input_sz, **kwargs):
        # This is a hack since TF never needs this one, there is not a general constructor param, so shoehorn
        kwargs['dsz'] = input_sz
        return create_seq2seq_encoder(**kwargs)

    def init_decoder(self, tgt_embedding, **kwargs):
        return create_seq2seq_decoder(tgt_embedding, **kwargs)

    def encode(self, input, lengths):
        """

        :param input:
        :param lengths:
        :return:
        """
        embed_in_seq = self.embed(input)
        return self.encoder(embed_in_seq, lengths)

    def decode(self, encoder_outputs, dst):
        return self.decoder(encoder_outputs, dst)

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

    def drop_inputs(self, key, x):
        v = self.dropin_values.get(key, 0)

        if not self.training or v == 0:
            return x

        mask_pad = x != Offsets.PAD
        mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(v).byte()
        x.masked_fill_(mask_pad & mask_drop, Offsets.UNK)
        return x

    def input_tensor(self, key, batch_dict, perm_idx):
        tensor = torch.from_numpy(batch_dict[key])
        tensor = self.drop_inputs(key, tensor)
        tensor = tensor[perm_idx]
        if self.gpu:
            tensor = tensor.cuda()
        return tensor

    def make_input(self, batch_dict):
        example = dict({})

        lengths = torch.from_numpy(batch_dict[self.src_lengths_key])
        lengths, perm_idx = lengths.sort(0, descending=True)

        if self.gpu:
            lengths = lengths.cuda()
        example['src_len'] = lengths
        for key in self.src_embeddings.keys():
            example[key] = self.input_tensor(key, batch_dict, perm_idx)

        if 'tgt' in batch_dict:
            tgt = torch.from_numpy(batch_dict['tgt'])
            example['dst'] = tgt[:, :-1]
            example['tgt'] = tgt[:, 1:]
            example['dst'] = example['dst'][perm_idx]
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
        # Return as B x T x H
        return output

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
                if isinstance(value, list):
                    example[k] = np.array([value[b]])
                else:
                    example[k] = value[b].reshape((1,) + value[b].shape)
            inputs = self.make_input(example)
            encoder_outputs = self.encode(inputs, inputs['src_len'])
            batch.append(self.decoder.predict_one(inputs[src_field], encoder_outputs, **kwargs)[0])
        return batch


@register_model(task='seq2seq', name=['default', 'attn'])
class Seq2SeqModel(EncoderDecoderModelBase):

    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        """This base model is extensible for attention and other uses.  It declares minimal fields allowing the
        subclass to take over most of the duties for drastically different implementations

        :param src_embeddings: (``dict``) A dictionary of PyTorchEmbeddings
        :param tgt_embedding: (``PyTorchEmbeddings``) A single PyTorchEmbeddings object
        :param kwargs:
        """
        super(Seq2SeqModel, self).__init__(src_embeddings, tgt_embedding, **kwargs)
