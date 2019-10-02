import logging
from baseline.pytorch.torchy import *
from baseline.pytorch.crf import *
from baseline.utils import Offsets, write_json
from baseline.model import TaggerModel
from baseline.model import register_model
import torch.autograd
import os

logger = logging.getLogger('baseline')


class TaggerModelBase(nn.Module, TaggerModel):

    def save(self, outname):
        torch.save(self, outname)
        basename, _ = os.path.splitext(outname)
        write_json(self.labels, basename + ".labels")

    def to_gpu(self):
        self.gpu = True
        self.cuda()
        if not self.use_crf:
            self.crit.cuda()
        return self

    @staticmethod
    def load(filename, **kwargs):
        device = kwargs.get('device')
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename, map_location=device)
        model.gpu = False if device == 'cpu' else model.gpu
        return model

    def __init__(self):
        super(TaggerModelBase, self).__init__()
        self.gpu = False

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
        model.activation_type = kwargs.get('activation', 'tanh')
        constraint = kwargs.get('constraint')

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
            model.crf = CRF(
                len(labels),
                (Offsets.GO, Offsets.EOS),
                batch_first=False,
                constraint=constraint
            )
        else:
            if constraint is not None:
                constraint = F.log_softmax(torch.zeros(constraint.shape).masked_fill(constraint, -1e4), dim=0)
                model.register_buffer('constraint', constraint.unsqueeze(0))
            else:
                model.constraint = None
            model.crit = SequenceCriterion(LossFn=nn.CrossEntropyLoss, avg='batch')
        logger.info(model)
        return model

    def drop_inputs(self, key, x):
        v = self.dropin_values.get(key, 0)

        if not self.training or v == 0:
            return x

        mask_pad = x != Offsets.PAD
        mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(v).to(mask_pad.dtype)
        x.masked_fill_(mask_pad & mask_drop, Offsets.UNK)
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
            if self.gpu:
                y = y.cuda()
            example_dict['y'] = y

        ids = batch_dict.get('ids')
        if ids is not None:
            ids = torch.from_numpy(ids)[perm_idx]
            if self.gpu:
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
        lengths = lengths.to(probv.device)
        if self.use_crf is True:
            preds, _ = self.crf.decode(probv, lengths)
        else:
            if self.constraint is not None:
                probv = F.log_softmax(probv, dim=-1)
                preds, _ = viterbi(probv, self.constraint, lengths, Offsets.GO, Offsets.EOS, norm=F.log_softmax)
            else:
                _, preds = torch.max(probv, 2)
        pred_list = []
        preds = preds.transpose(0, 1)
        for pred, sl in zip(preds, lengths):
            pred_list.append(pred[:sl])
        return pred_list

    def compute_loss(self, inputs):
        lengths = inputs['lengths']
        tags = inputs['y']

        probv = self.compute_unaries(inputs, lengths)
        batch_loss = 0.
        total_tags = 0.
        if self.use_crf is True:
            # Get tags as [T, B]
            tags = tags.transpose(0, 1)
            lengths = lengths.to(probv.device)
            batch_loss = torch.mean(self.crf.neg_log_loss(probv, tags.data, lengths))
        else:
            # Get batch (B, T)
            probv = probv.transpose(0, 1).contiguous()
            batch_loss = self.crit(probv, tags)

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
        logger.info('RNN [%s]' % rnntype)
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
        filtsz = kwargs.get('wfiltsz', 5)
        activation_type = kwargs.get('activation_type', 'relu')
        hsz = int(kwargs['hsz'])
        self.encoder = ConvEncoderStack(input_sz, hsz, filtsz, pdrop, layers, activation_type)
        return hsz

    def encode(self, tbh, lengths):
        # bct
        bht = tbh.permute(1, 2, 0).contiguous()
        bht = self.encoder(bht)
        # bht -> tbh
        return bht.permute(2, 0, 1).contiguous()
