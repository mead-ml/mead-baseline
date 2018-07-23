from baseline.model import create_lang_model, LanguageModel
from baseline.dy.dynety import *


class BaseLanguageModel(LanguageModel, DynetModel):
    def __init__(self):
        super(BaseLanguageModel, self).__init__()

    def embed(self, input_):
        pass

    def output(self, input_):
        return [self._output(x) for x in input_]

    def forward(self, input_, state=None, train=True):
        input_ = self.embed(input_)
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


class WordLanguageModel(BaseLanguageModel):
    def __init__(
            self,
            embeddings, finetune=True,
            layers=1, hsz=650,
            dropout=None,
            **kwargs
    ):
        super(WordLanguageModel, self).__init__()
        embedding = embeddings['word']
        vsz = embedding.vsz + 1
        dsz = embedding.dsz
        self._embed = Embedding(vsz, dsz, self.pc, embedding_weight=embedding.weights, finetune=finetune, batched=True)
        self._rnn = dy.VanillaLSTMBuilder(layers, dsz, hsz, self.pc)
        self._output = Linear(vsz, hsz, self.pc, name="output")
        self.dropout = dropout

    def embed(self, input_):
        words, _ = input_
        return self._embed(words)

    def make_input(self, batch_dict, **kwargs):
        x = batch_dict['x'].T
        y = batch_dict['y'].T
        return (x, None), y

    @classmethod
    def create(cls, embeddings, **kwargs):
        model = cls(embeddings, **kwargs)
        print(model)
        return model


class CharCompLanguageModel(BaseLanguageModel):
    def __init__(
            self, embedding,
            finetune=True, use_words=False,
            layers=1, hsz=650,
            cmotsz=30, cfiltsz=[1, 2, 3, 4, 5, 6, 7],
            max_feat=200, nfeat_factor=50,
            gating='highway', num_gates=2,
            dropout=None,
            **kwargs
    ):
        super(CharCompLanguageModel, self).__init__()
        self.use_words = use_words
        vsz = embedding['word'].vsz + 1
        dsz = embedding['word'].dsz
        cvsz = embedding['char'].vsz + 1
        cdsz = embedding['char'].dsz
        self.c_embed = Embedding(
            cvsz, cdsz,
            self.pc,
            embedding_weight=embedding['char'].weights,
            finetune=finetune, dense=True, batched=True
        )
        if self.use_words:
            self.w_embed = Embedding(
                vsz, dsz,
                self.pc,
                embedding_weight=embedding['word'].weights,
                finetune=finetune, dense=False, batched=True, name="word-embeddings"
            )
        self._char_comp, cmotsz_total = self._create_char_comp(
            cfiltsz, cmotsz, cdsz, gating, num_gates, max_feat, nfeat_factor
        )
        if use_words:
            rnninsz = cmotsz_total + dsz
        else:
            rnninsz = cmotsz_total
        self._rnn = dy.VanillaLSTMBuilder(layers, rnninsz, hsz, self.pc)
        self._output = Linear(vsz, hsz, self.pc, name="output")
        self.dropout = dropout

    def make_input(self, batch_dict, **kwargs):
        x = batch_dict['x'].T
        xch = batch_dict['xch']
        # Change characters from [B, Ts, Tw] to [Ts, Tw, B] for easy lookup
        xch = np.transpose(xch, (1, 2, 0))
        y = batch_dict['y'].T
        return (x, xch), y

    def _create_char_comp(self, filtsz, cmotsz, cdsz, gate, num_gate, max_feat=200, nfeat_factor=None):
        if nfeat_factor is not None:
            cmotsz = [min(nfeat_factor * fsz, max_feat) for fsz in filtsz]
            cmotsz_total = sum(cmotsz)
        else:
            cmotsz_total = cmotsz * len(filtsz)
        self.parallel_conv = ParallelConv(filtsz, cmotsz, cdsz, self.pc)
        gate = HighwayConnection if gate.startswith('highway') else SkipConnection
        funcs = [Linear(cmotsz_total, cmotsz_total, self.pc, name="linear-{}".format(i)) for i in range(num_gate)]
        gating = gate(funcs, cmotsz_total, self.pc)

        def call(input_):
            x = self.parallel_conv(input_)
            return gating(x)

        return call, cmotsz_total

    def embed(self, input_):
        words, chars = input_
        c_embed = [self.c_embed(c) for c in chars]
        result = [self._char_comp(c) for c in c_embed]
        if self.use_words:
            words = self.w_embed(words)
            result = [dy.concatenate([r, w], d=0) for r, w in zip(result, words)]
        return result


    @classmethod
    def create(cls, embeddings, **kwargs):
        model = cls(embeddings, **kwargs)
        print(model)
        return model

BASELINE_LM_MODELS = {
    'default': WordLanguageModel.create,
    'convchar': CharCompLanguageModel.create
}


def create_model(embeddings, **kwargs):
    lm = create_lang_model(BASELINE_LM_MODELS, embeddings, **kwargs)
    return lm


#def load_model(modelname, **kwargs):
#    return load_lang_model(WordLanguageModel.load, modelname, **kwargs)
