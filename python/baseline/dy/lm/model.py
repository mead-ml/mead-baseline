from baseline.model import LanguageModel, register_model
from baseline.dy.dynety import *


@register_model(task='lm', name='default')
class BasicLanguageModel(DynetModel, LanguageModel):

    @classmethod
    def create(cls, embeddings, **kwargs):
        return cls(embeddings, **kwargs)

    def __init__(self, embeddings, layers=1, hsz=650, dropout=None, **kwargs):
        super(BasicLanguageModel, self).__init__(kwargs['pc'])
        self.tgt_key = kwargs.get('tgt_key')
        vsz = embeddings[self.tgt_key].vsz
        dsz = self.init_embed(embeddings)
        self._rnn = dy.VanillaLSTMBuilder(layers, dsz, hsz, self.pc)
        self._output = Linear(vsz, hsz, self.pc, name="output")
        self.dropout = dropout

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
        if train:
            if self.dropout is not None:
                self._rnn.set_dropout(self.dropout)
        else:
            self._rnn.disable_dropout()
        transduced, last_state = rnn_forward_with_state(self._rnn, input_, None, state)
        return transduced, last_state

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
