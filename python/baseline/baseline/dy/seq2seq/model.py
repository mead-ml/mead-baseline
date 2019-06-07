import logging
import numpy as np
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

logger = logging.getLogger('baseline')


class EncoderDecoderModelBase(DynetModel, EncoderDecoderModel):
    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        super(EncoderDecoderModelBase, self).__init__(kwargs['pc'])
        self.beam_sz = kwargs.get('beam', 1)
        src_dsz = self.init_embed(src_embeddings, tgt_embedding)
        self.src_lengths_key = kwargs.get('src_lengths_key')
        self.encoder = self.init_encoder(src_dsz, **kwargs)
        self.decoder = self.init_decoder(tgt_embedding, **kwargs)
        self.train = True

    @classmethod
    def create(cls, src_embeddings, tgt_embedding, **kwargs):
        model = cls(src_embeddings, tgt_embedding, **kwargs)
        logger.info(model)
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

    def init_embed(self, src_embeddings, tgt_embedding, **kwargs):
        dsz = 0
        self.src_embeddings = src_embeddings
        for embedding in self.src_embeddings.values():
            dsz += embedding.get_dsz()
        return dsz

    def embed(self, batch_dict):
        all_embeddings = []
        for k, embedding in self.src_embeddings.items():
            all_embeddings.append(embedding.encode(batch_dict[k], self.train))
        embeddings = dy.concatenate(all_embeddings, d=1)
        return embeddings

    def init_encoder(self, src_dsz, **kwargs):
        kwargs['dsz'] = src_dsz
        return create_seq2seq_encoder(**kwargs)

    def init_decoder(self, tgt_embedding, **kwargs):
        return create_seq2seq_decoder(tgt_embedding, **kwargs)

    def encode(self, embed_in, lengths):
        embed_in_seq = self.embed(embed_in)
        return self.encoder(embed_in_seq, lengths, self.train)

    def decode(self, encoder_output, dst):
        return self.decoder(encoder_output, dst, self.train)

    def make_input(self, batch_dict):
        example_dict = dict({})
        for k in self.src_embeddings.keys():
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

    def predict(self, batch_dict, **kwargs):
        batch = []
        src_field = self.src_lengths_key.split('_')[0]
        B = batch_dict[src_field].shape[0]
        for b in range(B):
            dy.renew_cg()
            example = {}
            for k, v in batch_dict.items():
                example[k] = v[b, np.newaxis]
            inputs = self.make_input(example)
            encoder_outputs = self.encode(inputs, inputs['src_len'])
            batch.append(self.decoder.predict_one(inputs['src'], encoder_outputs, **kwargs)[0])
        return batch


@register_model(task='seq2seq', name=['default', 'attn'])
class Seq2SeqModel(EncoderDecoderModelBase):
    def __init__(self, src_embeddings, tgt_embedding, **kwargs):
        super(Seq2SeqModel, self).__init__(src_embeddings, tgt_embedding, **kwargs)
