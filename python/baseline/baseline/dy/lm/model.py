import logging
from baseline.model import LanguageModel, register_model
from baseline.dy.dynety import *
from baseline.dy.transformer import TransformerEncoderStack, subsequent_mask

logger = logging.getLogger('baseline')


class LanguageModelBase(DynetModel, LanguageModel):

    @classmethod
    def create(cls, embeddings, **kwargs):
        model = cls(embeddings, **kwargs)
        logger.info(model)
        return model

    def __init__(self, embeddings, layers=1, hsz=650, dropout=None, **kwargs):
        super(LanguageModelBase, self).__init__(kwargs['pc'])
        self.tgt_key = kwargs.get('tgt_key')
        vsz = embeddings[self.tgt_key].vsz
        dsz = self.init_embed(embeddings)
        self.init_decode(dsz, layers=layers, hsz=hsz, **kwargs)
        self.init_output(vsz, hsz, **kwargs)
        self.dropout = dropout

    def init_decode(self, dsz, layers=1, hsz=650, **kwargs):
        pass

    def init_output(self, vsz, hsz=650, **kwargs):
        do_weight_tying = bool(kwargs.get('tie_weights', False))
        embed = self.embeddings[self.tgt_key]
        if do_weight_tying and hsz == embed.get_dsz():
            self._output = WeightShareLinear(embed.get_vsz(), embed.embeddings, self.pc, transform=squeeze_and_transpose, name=embed.pc.name())
        else:
            self._output = Linear(vsz, hsz, self.pc, name="output")

    def init_embed(self, embeddings):
        dsz = 0
        self.embeddings = embeddings
        for embedding in self.embeddings.values():
            dsz += embedding.get_dsz()
        return dsz

    def embed(self, batch_dict):
        all_embeddings_lists = []
        for k, embedding in self.embeddings.items():
            all_embeddings_lists.append(embedding.encode(batch_dict[k]))

        embedded = dy.concatenate(all_embeddings_lists, d=1)
        return embedded

    def make_input(self, batch_dict):
        example_dict = dict({})
        for key in self.embeddings.keys():
            example_dict[key] = batch_dict[key].T
        y = batch_dict.get('y')
        if y is not None:
            example_dict['y'] = y.T
        return example_dict

    def output(self, input_):
        return [self._output(x) for x in input_]

    def decode(self, input_, state, train):
        pass

    def forward(self, input_, state=None, train=True):
        input_ = self.embed(input_)
        transduced, last_state = self.decode(input_, state, train)
        output = self.output(transduced)
        return output, last_state

    def save(self, file_name):
        self.pc.save(file_name)
        return self

    def load(self, file_name):
        self.pc.populate(file_name)
        return self


@register_model(task='lm', name='default')
class RNNLanguageModel(LanguageModelBase):

    def __init__(self, embeddings, layers=1, hsz=650, dropout=None, **kwargs):
        self._rnn = None
        super(RNNLanguageModel, self).__init__(embeddings, layers, hsz, dropout, **kwargs)

    def init_decode(self, dsz, layers=1, hsz=650, **kwargs):
        self._rnn = dy.VanillaLSTMBuilder(layers, dsz, hsz, self.pc)

    def decode(self, input_, state, train):
        if train:
            if self.dropout is not None:
                self._rnn.set_dropout(self.dropout)
        else:
            self._rnn.disable_dropout()
        transduced, last_state = rnn_forward_with_state(self._rnn, input_, None, state)
        return transduced, last_state


@register_model(task='lm', name='transformer')
class TransformerLanguageModel(LanguageModelBase):
    def __init__(self, *args, **kwargs):
        super(TransformerLanguageModel, self).__init__(*args, **kwargs)

    def init_decode(self, dsz, layers=1, hsz=650, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.5))
        d_model = int(kwargs.get('d_model', hsz))
        num_heads = int(kwargs.get('num_heads', 4))
        self.proj_to_dsz = Linear(d_model, dsz, self.pc)
        self.transformer = TransformerEncoderStack(num_heads, d_model=d_model, pdrop=pdrop, scale=True, layers=layers, pc=self.pc)

    def decode(self, th_b, hidden, train):
        ht_b = dy.transpose(th_b)
        T = ht_b.dim()[0][1]
        ht_b = self.proj_to_dsz(ht_b)
        mask = subsequent_mask(T)
        output = self.transformer(ht_b, mask, train)
        output = [out for out in dy.transpose(output)]
        return output, None
