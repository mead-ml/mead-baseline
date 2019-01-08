import logging
from baseline.pytorch.torchy import *
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

    def init_embed(self, **kwargs):
        return EmbeddingsStack(self.embeddings, self.pdrop)

    def init_encoder(self, input_sz, **kwargs):
        pass

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        model = cls()
        model.lengths_key = kwargs.get('lengths_key')
        model.activation_type = kwargs.get('activation', 'tanh')
        model.embeddings = embeddings
        model.pdrop = float(kwargs.get('dropout', 0.5))
        model.dropin_values = kwargs.get('dropin', {})
        model.labels = labels

        embed_model = model.init_embed(**kwargs)
        transducer_model = model.init_encoder(embed_model.output_dim, **kwargs)

        use_crf = bool(kwargs.get('crf', False))
        constraint = kwargs.get('constraint')
        if constraint is not None:
            constraint = constraint.unsqueeze(0)

        if use_crf:
            decoder_model = CRF(len(labels), constraint_mask=constraint, batch_first=False)
        else:
            decoder_model = TaggerGreedyDecoder(len(labels), constraint_mask=constraint)

        model.layers = TagSequenceModel(len(labels), embed_model, transducer_model, decoder_model)
        logger.info(model.layers)
        return model

    def drop_inputs(self, key, x):
        v = self.dropin_values.get(key, 0)

        if not self.training or v == 0:
            return x

        mask_pad = x != Offsets.PAD
        mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(v).byte()
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

    def forward(self, input):
        return self.layers(input)

    def compute_loss(self, inputs):
        tags = inputs['y'].transpose(0, 1)
        lengths = inputs['lengths']
        unaries = self.layers.transduce(inputs)
        return self.layers.neg_log_loss(unaries, tags, lengths)

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
        unif = kwargs.get('unif', 0)
        hsz = int(kwargs['hsz'])
        weight_init = kwargs.get('weight_init', 'uniform')
        rnntype = kwargs.get('rnntype', 'blstm')
        Encoder = LSTMEncoder if rnntype == 'lstm' else BiLSTMEncoder
        return Encoder(input_sz, hsz, layers, pdrop, unif=unif, initializer=weight_init, output_fn=rnn_signal)


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
