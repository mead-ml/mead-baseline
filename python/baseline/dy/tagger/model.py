import dynet as dy
from baseline.model import (
    TaggerModel,
    load_tagger_model,
    create_tagger_model
)
import numpy as np
from baseline.dy.dynety import CRF, Linear, Embedding, DynetModel, ParallelConv, rnn_forward


class RNNTaggerModelModel(TaggerModel, DynetModel):
    def __init__(self, embeddings_set, labels, finetune=True, dense=False, dropout=0.5, layers=1, **kwargs):
        super(RNNTaggerModelModel, self).__init__()
        self._pc = dy.ParameterCollection()
        self.pdrop = dropout
        self.train = True
        word_vsz = len(embeddings_set['word'].vocab)
        word_dsz = embeddings_set['word'].dsz
        char_vsz = len(embeddings_set['char'].vocab)
        self.char_dsz = embeddings_set['char'].dsz
        self.vocab = {}
        self.vocab['word'] = embeddings_set['word'].vocab
        self.vocab['char'] = embeddings_set['char'].vocab
        self.word_embed = Embedding(word_vsz, word_dsz, self.pc, embeddings_set['word'].weights, finetune, dense, batched=False, name='word-embedding')
        self.char_embed = Embedding(char_vsz, self.char_dsz, self.pc, embeddings_set['char'].weights, True, dense, batched=True, name='char-embedding')
        self.labels = labels

        self.hsz = int(kwargs['hsz'])
        self.pdrop = kwargs.get('dropout', 0.5)
        self.rnntype = kwargs.get('rnntype', 'blstm')
        self.do_crf = bool(kwargs.get('crf', False))
        self.crf_mask = bool(kwargs.get('crf_mask', False))
        self.span_type = kwargs.get('span_type')
        nc = len(self.labels)

        if self.do_crf:
            vocab = labels if self.crf_mask else None
            self.crf = CRF(nc, pc=self.pc, idxs=(labels['<GO>'], labels['<EOS>']),
                           vocab=vocab, span_type=self.span_type)

        self.activation_type = kwargs.get('activation', 'tanh')

        # Now make a BLSTM
        # Now make a CRF
        self.char_word_sz, self.pool_chars = self._init_pool_chars(self.char_dsz, kwargs.get('cfiltsz', [3]), kwargs.get('wsz', 30))
        self.rnn = dy.BiRNNBuilder(layers, self.char_word_sz + word_dsz, self.hsz, self.pc, dy.VanillaLSTMBuilder)
        self.output = self._init_output(self.hsz, nc)

    def dropout(self, input_):
        if self.train:
            return dy.dropout(input_, self.pdrop)
        return input_

    @property
    def pc(self):
        return self._pc

    def __str__(self):
        str_ = super(RNNTaggerModelModel, self).__str__()
        return "Auto-batching: \n{}".format(str_)

    def make_input(self, batch_dict):
        x = batch_dict['x']
        xch = batch_dict['xch']
        y = batch_dict['y']
        lengths = batch_dict['lengths']
        ids = batch_dict['ids']
        return x, xch, lengths, y, ids

    def forward(self, input_, lengths):
        x_b, xch_b = input_
        x = x_b.T
        xch = xch_b.transpose(2, 1, 0)

        embed_words_list = self.word_embed(x)
        W, T, B = xch.shape
        xch = xch.reshape(W, -1)
        # W x (T x B)
        embed_chars_list = self.char_embed(xch)
        embed_chars_vec = dy.concatenate(embed_chars_list)
        embed_chars_vec = dy.reshape(embed_chars_vec, (W, self.char_dsz), T*B)
        # Back to T x W x B
        pooled_chars = self.pool_chars(embed_chars_vec, 1)
        pooled_chars = dy.reshape(pooled_chars, (self.char_word_sz, T), B)
        pooled_chars = dy.transpose(pooled_chars)
        embed = [self.dropout(dy.concatenate([embed_word, pooled_char])) for embed_word, pooled_char in zip(embed_words_list, pooled_chars)]
        exps = [self.output(out) for out in rnn_forward(self.rnn, embed)]

        return exps

    def predict(self, input_, lengths):
        dy.renew_cg()

        unaries = self.forward(input_, lengths)
        if self.do_crf is True:
            best_path, path_score = self.crf.decode(unaries)
        else:
            best_path = [np.argmax(x.npvalue(), axis=0) for x in unaries]
        B, T = input_[0].shape

        best_path = np.stack(best_path).reshape((T, B))
        best_path = best_path.transpose(1, 0)
        results = []

        for b in range(best_path.shape[0]):
            sentence = best_path[b, :lengths[b]]
            results.append(sentence)
        return results

    def loss(self, preds, y):
        if self.do_crf is True:
            return self.crf.neg_log_loss(preds, y.squeeze(0))
        else:

            element_loss = dy.pickneglogsoftmax
            errs = []

            for pred, y_i in zip(preds, y.T):
                err = element_loss(pred, y_i)
                errs += [err]
            return dy.esum(errs)

    def dropout(self, input_):
        if self.train:
            return dy.dropout(input_, self.pdrop)
        return input_

    def _init_output(self, input_dim, n_classes):
        return Linear(n_classes, input_dim, self.pc, name="output")

    def _init_pool_chars(self, dsz, filtsz, cmotsz, **kwargs):
        parallel_conv = ParallelConv(filtsz, cmotsz, dsz, self.pc)
        def call_pool(input_, _):
            return parallel_conv(input_)
        return len(filtsz) * cmotsz, call_pool

    @classmethod
    def create(cls, embeddings_set, labels, **kwargs):
        model = cls(embeddings_set, labels, **kwargs)
        print(model)
        return model

    def save(self, file_name):
        self.pc.save(file_name)
        return self

    def load(self, file_name):
        self.pc.populate(file_name)
        return self


BASELINE_TAGGER_MODELS = {
    'default': RNNTaggerModelModel.create,
}

BASELINE_TAGGER_LOADERS = {
    'default': RNNTaggerModelModel.load,
}


def create_model(labels, embeddings, **kwargs):
    return create_tagger_model(BASELINE_TAGGER_MODELS, embeddings, labels, **kwargs)


def load_model(modelname, **kwargs):
    return load_tagger_model(BASELINE_TAGGER_LOADERS, modelname, **kwargs)
