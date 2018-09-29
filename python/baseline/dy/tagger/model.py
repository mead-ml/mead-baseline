import dynet as dy
from baseline.model import (
    TaggerModel,
    load_tagger_model,
    create_tagger_model
)
import numpy as np
from baseline.dy.dynety import CRF, Linear, DynetModel, rnn_forward


class RNNTaggerModel(TaggerModel, DynetModel):
    def __init__(self, embeddings_set, pc, labels, dropout=0.5, layers=1, **kwargs):
        super(RNNTaggerModel, self).__init__(pc)
        self.pdrop = dropout
        self.labels = labels
        self.hsz = int(kwargs['hsz'])
        self.pdrop = kwargs.get('dropout', 0.5)
        self.rnntype = kwargs.get('rnntype', 'blstm')
        self.do_crf = bool(kwargs.get('crf', False))
        self.crf_mask = bool(kwargs.get('crf_mask', False))
        self.span_type = kwargs.get('span_type')
        self.lengths_key = kwargs.get('lengths_key')
        dsz = self._init_embed(embeddings_set)
        nc = len(self.labels)

        if self.do_crf:
            vocab = labels if self.crf_mask else None
            self.crf = CRF(nc, pc=self.pc, idxs=(labels['<GO>'], labels['<EOS>']),
                           vocab=vocab, span_type=self.span_type)

        self.activation_type = kwargs.get('activation', 'tanh')
        self.rnn = dy.BiRNNBuilder(layers, dsz, self.hsz, self.pc, dy.VanillaLSTMBuilder)
        self.output = self._init_output(self.hsz, nc)

    def dropout(self, input_):
        if self.train:
            return dy.dropout(input_, self.pdrop)
        return input_

    @property
    def pc(self):
        return self._pc

    def __str__(self):
        str_ = super(RNNTaggerModel, self).__str__()
        return "Auto-batching: \n{}".format(str_)

    def _init_embed(self, embeddings):
        dsz = 0
        self.embeddings = embeddings
        for embedding in self.embeddings.values():
            dsz += embedding.get_dsz()
        return dsz

    def make_input(self, batch_dict):
        example_dict = dict({})

        # This forces everything to (T, B, W) or (T, B)
        # The former we dont actually want, but its ok, we fix it in the embedding input
        # using a transpose(2, 0, 1)
        for k, embedding in self.embeddings.items():
            example_dict[k] = batch_dict[k].T

        lengths = batch_dict[self.src_lengths_key]
        example_dict['lengths'] = lengths

        y = batch_dict.get('y')
        if y is not None:
            example_dict['y'] = y

        ids = batch_dict.get('ids')
        if ids is not None:
            example_dict['ids'] = ids

        return example_dict

    def _embed(self, input_):

        all_embeddings_lists = []
        for k, embedding in self.embeddings.items():
            all_embeddings_lists += [embedding.encode(input_)]

        pooled_vecs = [pooled for pooled in zip(all_embeddings_lists)]
        return self.dropout(dy.concatenate(pooled_vecs))

    def compute_unaries(self, batch_dict):
        embed = self._embed(batch_dict)
        exps = [self.output(out) for out in rnn_forward(self.rnn, embed)]
        return exps

    def predict(self, batch_dict):
        dy.renew_cg()
        inputs = self.make_input(batch_dict)
        lengths = inputs['lengths']
        unaries = self.compute_unaries(batch_dict)
        if self.do_crf is True:
            best_path, path_score = self.crf.decode(unaries)
        else:
            best_path = [np.argmax(x.npvalue(), axis=0) for x in unaries]
        B, T = batch_dict[self.lengths_key].shape
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
    'default': RNNTaggerModel.create,
}

BASELINE_TAGGER_LOADERS = {
    'default': RNNTaggerModel.load,
}


def create_model(labels, embeddings, **kwargs):
    return create_tagger_model(BASELINE_TAGGER_MODELS, embeddings, labels, **kwargs)


def load_model(modelname, **kwargs):
    return load_tagger_model(BASELINE_TAGGER_LOADERS, modelname, **kwargs)
