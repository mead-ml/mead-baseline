from baseline.model import ClassifierModel, load_classifier_model, create_classifier_model
from baseline.pytorch.torchy import *
from baseline.pytorch.embeddings import pytorch_embeddings, PyTorchCharConvEmbeddings, PyTorchWordEmbeddings
from baseline.utils import listify
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


class ClassifierModelBase(nn.Module, ClassifierModel):

    def __init__(self):
        super(ClassifierModelBase, self).__init__()

    @classmethod
    def load(cls, outname, **kwargs):
        model = torch.load(outname)
        return model

    def save(self, outname):
        print('saving %s' % outname)
        torch.save(self, outname)

    @classmethod
    def create(cls, embeddings, labels, **kwargs):

        model = cls()
        model.pdrop = kwargs.get('pdrop', 0.5)

        model.lengths_key = kwargs.get('lengths_key')
        if model.lengths_key is None:
            if 'word' in embeddings:
                model.lengths_key = 'word'
            elif 'x' in embeddings:
                model.lengths_key = 'x'

        if model.lengths_key is not None:
            # This allows user to short-hand the field to use
            if not model.lengths_key.endswith('_lengths'):
                model.lengths_key += '_lengths'
        input_sz = model._init_embed(embeddings)
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.labels = labels
        model.log_softmax = nn.LogSoftmax(dim=1)
        pool_dim = model._init_pool(input_sz, **kwargs)
        stacked_dim = model._init_stacked(pool_dim, **kwargs)
        model._init_output(stacked_dim, len(labels))
        print(model)
        return model

    def cuda(self, device=None):
        super(ClassifierModelBase, self).cuda(device=device)
        for emb in self.embeddings.values():
            emb.cuda(device)

    def create_loss(self):
        return nn.NLLLoss()

    def make_input(self, batch_dict):
        """Transform a `batch_dict` into something usable in this model

        :param batch_dict: (``dict``) A dictionary containing all inputs to the embeddings for this model
        :param do_dropout: (``bool``) Should we do dropout.  Defaults to False
        :return:
        """
        example_dict = dict({})
        for key in self.embeddings.keys():
            example_dict[key] = torch.from_numpy(batch_dict[key])
            if self.gpu:
                example_dict[key] = example_dict[key].cuda()

        # Allow us to track a length, which is needed for BLSTMs
        if self.lengths_key is not None:
            example_dict['lengths'] = torch.from_numpy(batch_dict[self.lengths_key])
            if self.gpu:
                example_dict['lengths'] = example_dict['lengths'].cuda()

        y = batch_dict.get('y')
        if y is not None:
            y = torch.from_numpy(y)
            if self.gpu is not None:
                y = y.cuda()
            example_dict['y'] = y

        return example_dict

    def forward(self, input):
        # BxTxC
        embeddings = self._embed(input)
        pooled = self._pool(embeddings, input['lengths'])
        stacked = self._stacked(pooled)
        return self.output(stacked)

    def _embed(self, input):
        all_embeddings = []
        for k, embedding in self.embeddings.items():
            all_embeddings += [embedding.encode(input[k])]
        return torch.cat(all_embeddings, 2)

    def classify(self, batch_dict):
        examples = self.make_input(batch_dict)

        with torch.no_grad():
            probs = self(examples).exp()
            probs.div_(torch.sum(probs))
            results = []
            batchsz = probs.size(0)
            for b in range(batchsz):
                outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
                results.append(outcomes)
        return results

    def get_labels(self):
        return self.labels

    def get_vocab(self, name='word'):
        return self.vocab.get(name)

    def _pool(self, embeddings, lengths):
        pass

    def _stacked(self, pooled):
        if self.stacked is None:
            return pooled
        return self.stacked(pooled)

    def _init_embed(self, embeddings, **kwargs):
        self.embeddings = EmbeddingsContainer()
        input_sz = 0
        for k, embedding in embeddings.items():
            DefaultType = PyTorchCharConvEmbeddings if k == 'char' else PyTorchWordEmbeddings
            self.embeddings[k] = pytorch_embeddings(embeddings[k], DefaultType)
            input_sz += self.embeddings[k].get_dsz()
        return input_sz

    def _init_stacked(self, input_dim, **kwargs):
        hszs = listify(kwargs.get('hsz', []))
        if len(hszs) == 0:
            self.stacked = None
            return input_dim
        self.stacked = nn.Sequential()
        layers = []
        in_layer_sz = input_dim
        for i, hsz in enumerate(hszs):
            layers.append(nn.Linear(in_layer_sz, hsz))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.pdrop))
            in_layer_sz = hsz
        append2seq(self.stacked, layers)
        return in_layer_sz

    def _init_output(self, input_dim, nc):
        self.output = nn.Sequential()
        append2seq(self.output, (
            nn.Linear(input_dim, nc),
            nn.LogSoftmax(dim=1)
        ))

    def _init_pool(self, dsz, **kwargs):
        pass


class ConvModel(ClassifierModelBase):

    def __init__(self):
        super(ConvModel, self).__init__()

    def _init_pool(self, dsz, **kwargs):
        filtsz = kwargs['filtsz']
        cmotsz = kwargs['cmotsz']
        self.parallel_conv = ParallelConv(dsz, cmotsz, filtsz, "relu", self.pdrop)
        return self.parallel_conv.outsz

    def _pool(self, btc, lengths):
        embeddings = btc.transpose(1, 2).contiguous()
        return self.parallel_conv(embeddings)


class LSTMModel(ClassifierModelBase):

    def __init__(self):
        super(LSTMModel, self).__init__()

    def _init_pool(self, dsz, **kwargs):
        unif = kwargs.get('unif')
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        self.lstm = nn.LSTM(dsz, hsz, 1, bias=True, dropout=self.pdrop)
        if unif is not None:
            for weight in self.lstm.parameters():
                weight.data.uniform_(-unif, unif)
        return hsz

    def _pool(self, embeddings, lengths):

        embeddings = embeddings.transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths.tolist())
        output, hidden = self.lstm(packed)
        hidden = hidden[0].view(hidden[0].shape[1:])
        return hidden

    def make_input(self, batch_dict):

        x = batch_dict['x']
        xch = batch_dict.get('xch')
        y = batch_dict.get('y')
        lengths = batch_dict['lengths']
        lengths, perm_idx = lengths.sort(0, descending=True)
        x = x[perm_idx]
        if xch is not None:
            xch = xch[perm_idx]
        if y is not None:
            y = y[perm_idx]
        if self.gpu:
            x = x.cuda()
            if xch is not None:
                xch = xch.cuda()
            if y is not None:
                y = y.cuda()

        if y is not None:
            y = y.contiguous()

        return x, xch, lengths, y


class NBowBase(ClassifierModelBase):

    def _init__(self):
        super(NBowBase, self)._init__()

    def _init_pool(self, dsz, **kwargs):
        return dsz

    def _init_stacked(self, input_dim, **kwargs):
        kwargs['hsz'] = kwargs.get('hsz', [100])
        return super(NBowBase, self)._init_stacked(input_dim, **kwargs)


class NBowModel(NBowBase):

    def __init__(self):
        super(NBowModel, self).__init__()

    def _pool(self, embeddings):
        return torch.mean(embeddings, 1, False)


class NBowMaxModel(NBowBase):
    def __init__(self):
        super(NBowMaxModel, self).__init__()

    def _pool(self, embeddings, lengths):
        dmax, _ = torch.max(embeddings, 1, False)
        return dmax


# These define the possible models for this backend
BASELINE_CLASSIFICATION_MODELS = {
    'default': ConvModel.create,
    'lstm': LSTMModel.create,
    'nbow': NBowModel.create,
    'nbowmax': NBowMaxModel.create
}
BASELINE_CLASSIFICATION_LOADERS = {
    'default': ConvModel.load,
    'lstm': LSTMModel.load,
    'nbow': NBowModel.load,
    'nbowmax': NBowMaxModel.create
}


def create_model(embeddings, labels, **kwargs):
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, embeddings, labels, **kwargs)


def load_model(outname, **kwargs):
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADERS, outname, **kwargs)
