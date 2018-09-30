from baseline.model import create_lang_model, load_lang_model, LanguageModel
from baseline.dy.dynety import *


class BasicLanguageModel(DynetModel, LanguageModel):

    @classmethod
    def create(cls, embeddings, **kwargs):
        return cls(embeddings, **kwargs)

    def __init__(self, embeddings, layers=1, hsz=650, dropout=None, **kwargs):
        super(BasicLanguageModel, self).__init__(kwargs['pc'])
        self.tgt_key = kwargs.get('tgt_key')
        vsz = embeddings[self.tgt_key].vsz
        dsz = self._init_embed(embeddings)
        self._rnn = dy.VanillaLSTMBuilder(layers, dsz, hsz, self.pc)
        self._output = Linear(vsz, hsz, self.pc, name="output")
        self.dropout = dropout

    def _init_embed(self, embeddings):
        dsz = 0
        self.embeddings = embeddings
        for embedding in self.embeddings.values():
            dsz += embedding.get_dsz()
        return dsz

    def _embed(self, batch_dict):
        all_embeddings_lists = []
        for k, embedding in self.embeddings.items():
            all_embeddings_lists += [embedding.encode(batch_dict[k])]

        embed = dy.concatenate(all_embeddings_lists, d=1)
        return embed

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

    def forward(self, input_, state=None, train=True):
        input_ = self._embed(input_)
        if train:
            self._rnn.set_dropout(self.dropout)
        else:
            self._rnn.disable_dropout()
        transduced, last_state = rnn_forward_with_state(self._rnn, input_, None, state)
        output = self.output(transduced)
        return output, last_state

    def save(self, file_name):
        self.pc.save(file_name)
        return self

    def load(self, file_name):
        self.pc.populate(file_name)
        return self


BASELINE_LM_MODELS = {
    'default': BasicLanguageModel.create,
}


def create_model(embeddings, **kwargs):
    lm = create_lang_model(BASELINE_LM_MODELS, embeddings, **kwargs)
    return lm


def load_model(modelname, **kwargs):
    return load_lang_model(BasicLanguageModel.load, modelname, **kwargs)
