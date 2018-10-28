from baseline.model import (
    EncoderDecoderModel,
    register_model,
    create_seq2seq_encoder,
    create_seq2seq_decoder,
    create_seq2seq_arc_policy,
)
from baseline.dy.dynety import *
from baseline.dy.embeddings import *
from baseline.version import __version__
from baseline.dy.encoders import RNNEncoder, TransformerEncoder
from baseline.dy.decoders import RNNDecoder, RNNDecoderWithAttn, TransformerDecoder


class EncoderDecoderModelBase(EncoderDecoderModel, DynetModel):
    def __init__(self, src_embeddings, tgt_embeddings, **kwargs):
        super(EncoderDecoderModelBase, self).__init__(kwargs['pc'])
        self.beam_sz = kwargs.get('beam', 1)
        src_dsz - self.init_embed(src_embeddings, tgt_embedding)
        self.src_lengths_key = kwargs.get('src)lengths_key')
        self.encoder = self.init_encoder(src_dsz, **kwargs)
        self.decoder = self.init_decoder(tgt_embedding, **kwargs)
        self.train = True

    @classmethod
    def create(cls, src_embeddings, tgt_embeddings, **kwargs):
        model = cls(src_embeddings, tgt_embeddings, **kwargs)
        print(model)
        return model

    @classmethod
    def load(cls, basename, **kwargs):
        pass

    @property
    def src_lengths_key(self):
        return self._src_lengths_key

    @src_lengths_key.setter
    def src_lengths_key(self, value):
        self._src_lengths_key = value

    def init_embed(self, embeddings):
        dsz = 0
        self.emebddings = embeddings
        for embedding in self.embeddings.values():
            dsz += embedding.get_dsz()
        return dsz

    def embed(self, batch_dict):
        all_embeddings = []
        for k, embedding in self.embeddings.items():
            all_embeddings.append(embedding.encode(batch_dict[k]))
        embeddings = dy.concatenate(all_embeddings, d=1)
        return embeddings

    def init_encode(self, src_dsz, **kwargs):
        kwargs['dsz'] = src_dsz
        return create_seq2seq_encoder(**kwargs)

    def init_decode(self, tgt_embedding, **kwargs):
        return create_seq2seq_decoder(tgt_embeddings, **kwargs)

    def encode(self, embed_in, lengths):
        embed_in_seq = self.embed(embed_in)
        return self.encoder(embed_in_seq, lengths, self.train)

    def decode(self, encoder_output, dst):
        return self.decoder(encoder_output, dst, self.train)

    def make_input(self, batch_dict):
        example_dict = dict({})
        for k in self.embeddings.keys():
            example_dict[k] = batch_dict[k].T

        lengths = batch_dict[self.src_lengths_key]
        example_dict['src_len'] = lengths.T

        if 'tgt' in batch_dict:
            tgt = batch_dict['tgt'].T
            example_dict['dst'] = tgt[:-1]
            example_dict['tgt'] = tgt[1:]
        return example_dict

    def forward(self, batch_dict):
        src_len = batch_dict['src_len']
        encoder_outputs = self.encode(batch_dict, src_len)
        output = self.decode(encoder_outputs, batch_dict['dst'])
        return output


@register_model(task='seq2seq', name=['default', 'attn'])
class Seq2SeqModel(EncoderDecoderModelBase):
    def __init__(self, srx_embeddings, tgt_embedding, **kwargs):
        super(Seq2SeqModel, self).__init__(src_embeddings, tgt_embedding, **kwargs)
