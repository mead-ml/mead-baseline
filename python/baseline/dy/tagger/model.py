import dynet as dy
from baseline.model import (
    TaggerModel,
    register_model
)
import numpy as np
from baseline.dy.dynety import CRF, Linear, DynetModel, rnn_forward


@register_model(task='tagger', name='default')
class TaggerModelBase(DynetModel, TaggerModel):
    def __init__(self, embeddings_set, labels, **kwargs):
        super(TaggerModelBase, self).__init__(kwargs['pc'])
        self.labels = labels
        self.hsz = int(kwargs['hsz'])
        self.pdrop = kwargs.get('dropout', 0.5)
        self.rnntype = kwargs.get('rnntype', 'blstm')
        self.do_crf = bool(kwargs.get('crf', False))
        self.crf_mask = bool(kwargs.get('crf_mask', False))
        self.span_type = kwargs.get('span_type')
        self.lengths_key = kwargs.get('lengths_key')
        dsz = self.init_embed(embeddings_set)
        nc = len(self.labels)

        if self.do_crf:
            vocab = labels if self.crf_mask else None
            self.crf = CRF(nc, pc=self.pc, idxs=(labels['<GO>'], labels['<EOS>']),
                           vocab=vocab, span_type=self.span_type)

        self.activation_type = kwargs.get('activation', 'tanh')
        self.init_encoder(dsz, **kwargs)
        self.output = self.init_output(self.hsz, nc)

    def init_encoder(self, input_sz, **kwargs):
        pass

    def dropout(self, input_):
        if self.train:
            return dy.dropout(input_, self.pdrop)
        return input_

    def __str__(self):
        str_ = super(TaggerModelBase, self).__str__()
        return "Auto-batching: \n{}".format(str_)

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
        embed_list = [self.dropout(e) for e in embedded]
        return embed_list

    def make_input(self, batch_dict):
        example_dict = dict({})

        for k, embedding in self.embeddings.items():
            example_dict[k] = batch_dict[k].T

        lengths = batch_dict[self.lengths_key]
        example_dict['lengths'] = lengths

        y = batch_dict.get('y')
        if y is not None:
            example_dict['y'] = y

        ids = batch_dict.get('ids')
        if ids is not None:
            example_dict['ids'] = ids

        return example_dict

    def compute_unaries(self, batch_dict):
        embed_list = self._embed(batch_dict)
        exps = self.encode(embed_list)
        return exps

    def encode(self, embed_list):
        pass

    def predict(self, batch_dict):
        dy.renew_cg()
        inputs = self.make_input(batch_dict)
        lengths = inputs['lengths']
        unaries = self.compute_unaries(inputs)
        if self.do_crf is True:
            best_path, path_score = self.crf.decode(unaries)
        else:
            best_path = [np.argmax(x.npvalue(), axis=0) for x in unaries]
        # TODO: RN using autobatching, so none of this is really useful
        # If we want to support batching in this function we have to either loop over the batch
        # or we can just simplify all this code here
        best_path = np.stack(best_path).reshape(-1, 1)  # (T, B)

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
                errs.append(err)
            return dy.esum(errs)

    def dropout(self, input_):
        if self.train:
            return dy.dropout(input_, self.pdrop)
        return input_

    def init_output(self, input_dim, n_classes):
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


class RNNTaggerModel(TaggerModelBase):

    def __init__(self, embeddings_set, labels, **kwargs):
        super(RNNTaggerModel, self).__init__(embeddings_set, labels, **kwargs)

    def init_encoder(self, input_sz, **kwargs):
        layers = kwargs.get('layers', 1)
        self.rnn = dy.BiRNNBuilder(layers, input_sz, self.hsz, self.pc, dy.VanillaLSTMBuilder)

    def encode(self, embed_list):
        return [self.output(out) for out in rnn_forward(self.rnn, embed_list)]