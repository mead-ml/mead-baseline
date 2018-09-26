from baseline.pytorch.torchy import *
from baseline.pytorch.crf import *
from baseline.model import TaggerModel, create_tagger_model, load_tagger_model
import torch.autograd


class RNNTaggerModel(nn.Module, TaggerModel):

    def save(self, outname):
        torch.save(self, outname)

    def to_gpu(self):
        self.gpu = True
        self.cuda()
        self.crit.cuda()
        return self

    @staticmethod
    def load(outname, **kwargs):
        model = torch.load(outname)
        return model

    def __init__(self):
        super(RNNTaggerModel, self).__init__()

    def _init_embed(self, embeddings, **kwargs):
        self.embeddings = EmbeddingsContainer()
        input_sz = 0
        for k, embedding in embeddings.items():
            self.embeddings[k] = embedding
            input_sz += embedding.get_dsz()
        return input_sz

    def _embed(self, input):
        all_embeddings = []
        for k, embedding in self.embeddings.items():
            all_embeddings += [embedding.encode(input[k])]
        return torch.cat(all_embeddings, 2)

    @classmethod
    def create(cls, labels, embeddings, **kwargs):
        model = cls()
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

        hsz = int(kwargs['hsz'])
        model.proj = bool(kwargs.get('proj', False))
        model.use_crf = bool(kwargs.get('crf', False))
        model.crf_mask = bool(kwargs.get('crf_mask', False))
        model.span_type = kwargs.get('span_type')
        model.activation_type = kwargs.get('activation', 'tanh')
        nlayers = int(kwargs.get('layers', 1))
        rnntype = kwargs.get('rnntype', 'blstm')
        model.gpu = False
        print('RNN [%s]' % rnntype)

        pdrop = float(kwargs.get('dropout', 0.5))
        model.labels = labels
        input_sz = model._init_embed(embeddings, **kwargs)
        model.dropout = nn.Dropout(pdrop)
        model.rnn = LSTMEncoder(input_sz, hsz, rnntype, nlayers, pdrop)
        out_hsz = model.rnn.outsz
        model.decoder = nn.Sequential()
        if model.proj is True:
            append2seq(model.decoder, (
                pytorch_linear(out_hsz, hsz),
                pytorch_activation(model.activation_type),
                nn.Dropout(pdrop),
                pytorch_linear(hsz, len(model.labels))
            ))
        else:
            append2seq(model.decoder, (
                pytorch_linear(out_hsz, len(model.labels)),
            ))

        if model.use_crf:
            if model.crf_mask:
                assert model.span_type is not None, "A crf mask cannot be used without providing `span_type`"
                model.crf = CRF(
                    len(labels),
                    (model.labels.get("<GO>"), model.labels.get("<EOS>")), batch_first=False,
                    vocab=model.labels, span_type=model.span_type, pad_idx=model.labels.get("<PAD>")
                )
            else:
                model.crf = CRF(len(labels), (model.labels.get("<GO>"), model.labels.get("<EOS>")), batch_first=False)
        model.crit = SequenceCriterion(LossFn=nn.CrossEntropyLoss)
        print(model)
        return model

    def make_input(self, batch_dict):
        example_dict = dict({})

        lengths = torch.from_numpy(batch_dict[self.lengths_key])
        lengths, perm_idx = lengths.sort(0, descending=True)

        if self.gpu:
            lengths = lengths.cuda()
        example_dict['lengths'] = lengths
        for key in self.embeddings.keys():
            tensor = torch.from_numpy(batch_dict[key])
            tensor = tensor[perm_idx]
            example_dict[key] = tensor.transpose(0, 1).contiguous()
            if self.gpu:
                example_dict[key] = example_dict[key].cuda()

        y = batch_dict.get('y')
        if y is not None:
            y = torch.from_numpy(y)[perm_idx]
            if self.gpu is not None:
                y = y.cuda()
            example_dict['y'] = y

        ids = batch_dict.get('ids')
        if ids is not None:
            ids = torch.from_numpy(ids)[perm_idx]
            if self.gpu is not None:
                ids = ids.cuda()
            example_dict['ids'] = ids
        return example_dict

    def compute_unaries(self, inputs, lengths):
        words_over_time = self._embed(inputs)
        dropped = self.dropout(words_over_time)
        # output = (T, B, H)
        output = self.rnn(dropped, lengths)
        # stack (T x B, H)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), -1))

        # back to T x B x H
        decoded = decoded.view(output.size(0), output.size(1), -1)

        return decoded

    def forward(self, input):
        lengths = input['lengths']
        probv = self.compute_unaries(input, lengths)
        if self.use_crf is True:
            lengths = lengths.cuda()
            preds, _ = self.crf.decode(probv, lengths)
        else:
            # Get batch (B, T)
            probv = probv.transpose(0, 1)
            preds = []
            for pij, sl in zip(probv, lengths):
                _, unary = torch.max(pij[:sl], 1)
                preds.append(unary.data)

        return preds

    def compute_loss(self, inputs):
        lengths = inputs['lengths']
        tags = inputs['y']

        probv = self.compute_unaries(inputs, lengths)
        batch_loss = 0.
        total_tags = 0.
        if self.use_crf is True:
            # Get tags as [T, B]
            tags = tags.transpose(0, 1)
            lengths = lengths.cuda()
            batch_loss = torch.mean(self.crf.neg_log_loss(probv, tags.data, lengths))
        else:
            # Get batch (B, T)
            probv = probv.transpose(0, 1)
            for pij, gold, sl in zip(probv, tags, lengths):
                unary = pij[:sl]
                gold_tags = gold[:sl]
                total_tags += len(gold_tags)
                batch_loss += self.crit(unary, gold_tags)
            batch_loss /= probv.shape[0]

        return batch_loss

    def get_vocab(self, vocab_type='word'):
        return self.word_vocab if vocab_type == 'word' else self.char_vocab

    def get_labels(self):
        return self.labels

    def predict(self, batch_dict):
        x = batch_dict['word']
        xch = batch_dict['char']
        lengths = batch_dict['lengths']
        return predict_seq_bt(self, x, xch, lengths)

BASELINE_TAGGER_MODELS = {
    'default': RNNTaggerModel.create,
}

BASELINE_TAGGER_LOADERS = {
    'default': RNNTaggerModel.load,
}


def create_model(labels, embeddings, **kwargs):
    return create_tagger_model(BASELINE_TAGGER_MODELS, labels, embeddings, **kwargs)


def load_model(modelname, **kwargs):
    return load_tagger_model(BASELINE_TAGGER_LOADERS, modelname, **kwargs)
