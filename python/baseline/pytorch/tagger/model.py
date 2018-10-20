from baseline.pytorch.torchy import *
from baseline.pytorch.crf import *
from baseline.model import TaggerModel
from baseline.model import register_model
import torch.autograd
import os


class TaggerModelBase(nn.Module, TaggerModel):

    PAD = 0
    UNK = 1

    def save(self, outname):
        torch.save(self, outname)

    def to_gpu(self):
        self.gpu = True
        self.cuda()
        self.crit.cuda()
        return self

    @staticmethod
    def load(filename, **kwargs):
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename)
        return model

    def __init__(self):
        super(TaggerModelBase, self).__init__()

    def init_embed(self, embeddings, **kwargs):
        self.embeddings = EmbeddingsContainer()
        input_sz = 0
        for k, embedding in embeddings.items():
            self.embeddings[k] = embedding
            input_sz += embedding.get_dsz()

        self.vdrop = bool(kwargs.get('variational_dropout', False))
        pdrop = float(kwargs.get('dropout', 0.5))
        if self.vdrop:
            self.dropout = VariationalDropout(pdrop)
        else:
            self.dropout = nn.Dropout(pdrop)
        return input_sz

    def embed(self, input):
        all_embeddings = []
        for k, embedding in self.embeddings.items():
            all_embeddings.append(embedding.encode(input[k]))
        return self.dropout(torch.cat(all_embeddings, 2))

    def init_encoder(self, input_sz, **kwargs):
        pass

    def encode(self, words_over_time, lengths):
        pass

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        model = cls()
        model.lengths_key = kwargs.get('lengths_key')
        model.proj = bool(kwargs.get('proj', False))
        model.use_crf = bool(kwargs.get('crf', False))
        model.crf_mask = bool(kwargs.get('crf_mask', False))
        model.span_type = kwargs.get('span_type')
        model.activation_type = kwargs.get('activation', 'tanh')

        model.gpu = False
        pdrop = float(kwargs.get('dropout', 0.5))
        model.dropin_values = kwargs.get('dropin', {})
        model.labels = labels

        input_sz = model.init_embed(embeddings, **kwargs)
        hsz = model.init_encoder(input_sz, **kwargs)

        model.decoder = nn.Sequential()
        if model.proj is True:
            append2seq(model.decoder, (
                pytorch_linear(hsz, hsz),
                pytorch_activation(model.activation_type),
                nn.Dropout(pdrop),
                pytorch_linear(hsz, len(model.labels))
            ))
        else:
            append2seq(model.decoder, (
                pytorch_linear(hsz, len(model.labels)),
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

    def drop_inputs(self, key, x):
        v = self.dropin_values.get(key, 0)

        if not self.training or v == 0:
            return x

        mask_pad = x != TaggerModelBase.PAD
        mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(v).byte()
        x.masked_fill_(mask_pad & mask_drop, TaggerModelBase.UNK)
        return x

    def input_tensor(self, key, batch_dict, perm_idx):
        tensor = torch.from_numpy(batch_dict[key])
        tensor = self.drop_inputs(key, tensor)
        tensor = tensor[perm_idx]
        tensor = tensor.transpose(0, 1).contiguous()
        if self.gpu:
            tensor = tensor.cuda()
        return tensor

    def make_input(self, batch_dict):
        example_dict = dict({})
        lengths = torch.from_numpy(batch_dict[self.lengths_key])
        lengths, perm_idx = lengths.sort(0, descending=True)

        if self.gpu:
            lengths = lengths.cuda()
        example_dict['lengths'] = lengths
        for key in self.embeddings.keys():
            example_dict[key] = self.input_tensor(key, batch_dict, perm_idx)

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
        words_over_time = self.embed(inputs)
        # output = (T, B, H)
        output = self.encode(words_over_time, lengths)
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
        inputs = self.make_input(batch_dict)
        return self(inputs)


@register_model(task='tagger', name='default')
class RNNTaggerModel(TaggerModelBase):

    def __init__(self):
        super(RNNTaggerModel, self).__init__()

    def init_encoder(self, input_sz, **kwargs):
        layers = int(kwargs.get('layers', 1))
        pdrop = float(kwargs.get('dropout', 0.5))
        rnntype = kwargs.get('rnntype', 'blstm')
        print('RNN [%s]' % rnntype)
        unif = kwargs.get('unif', 0)
        hsz = int(kwargs['hsz'])
        weight_init = kwargs.get('weight_init', 'uniform')
        self.encoder = LSTMEncoder(input_sz, hsz, rnntype, layers, pdrop, unif=unif, initializer=weight_init)
        return hsz

    def encode(self, words_over_time, lengths):
        return self.encoder(words_over_time, lengths)


@register_model(task='tagger', name='cnn')
class CNNTaggerModel(TaggerModelBase):

    def __init__(self):
        super(CNNTaggerModel, self).__init__()

    def init_encoder(self, input_sz, **kwargs):
        layers = int(kwargs.get('layers', 1))
        pdrop = float(kwargs.get('dropout', 0.5))
        filtsz = kwargs.get('filtsz', 5)
        activation_type = kwargs.get('activation_type', 'relu')
        hsz = int(kwargs['hsz'])
        self.encoder = ConvEncoderStack(input_sz, hsz, filtsz, activation_type, pdrop, layers)

    def encode(self, tbh, lengths):
        # tbh
        bht = tbh.permute([1, 0, 2]).contiguous()
        bht = self.encoder(bht)
        return bht.transpose([1, 2, 0]).contiguous()
