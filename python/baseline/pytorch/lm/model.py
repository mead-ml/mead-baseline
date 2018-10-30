from baseline.pytorch.torchy import *
from baseline.pytorch.transformer import TransformerEncoderStack, subsequent_mask
from baseline.model import LanguageModel, register_model
import torch.autograd
import os


class LanguageModelBase(nn.Module, LanguageModel):
    def __init__(self):
        super(LanguageModelBase, self).__init__()

    def save(self, outname):
        torch.save(self, outname)

    def create_loss(self):
        return SequenceCriterion(LossFn=nn.CrossEntropyLoss)

    @classmethod
    def load(cls, filename, **kwargs):
        if not os.path.exists(filename):
            filename += '.pyt'
        model = torch.load(filename)
        return model

    def init_hidden(self, batchsz):
        return None

    def make_input(self, batch_dict):
        example_dict = dict({})
        for key in self.embeddings.keys():
            example_dict[key] = torch.from_numpy(batch_dict[key])
            if self.gpu:
                example_dict[key] = example_dict[key].cuda()

        y = batch_dict.get('y')
        if y is not None:
            y = torch.from_numpy(y)
            if self.gpu is not None:
                y = y.cuda()
            example_dict['y'] = y
        return example_dict

    def embed(self, input):
        all_embeddings = []
        for k, embedding in self.embeddings.items():
            all_embeddings.append(embedding.encode(input[k]))
        embedded = torch.cat(all_embeddings, 2)
        return self.embed_dropout(embedded)

    def init_embed(self, embeddings, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.5))
        self.embed_dropout = nn.Dropout(pdrop)
        self.embeddings = EmbeddingsContainer()
        input_sz = 0
        for k, embedding in embeddings.items():
            self.embeddings[k] = embedding
            input_sz += embedding.get_dsz()
        return input_sz

    def init_decode(self, vsz, **kwargs):
        pass

    def decode(self, emb, hidden):
        pass

    @classmethod
    def create(cls, embeddings, **kwargs):

        lm = cls()
        lm.gpu = kwargs.get('gpu', True)
        lm.hsz = int(kwargs['hsz'])
        lm.layers = int(kwargs.get('layers', 1))
        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination ')

        lm.dsz = lm.init_embed(embeddings, **kwargs)
        lm.init_decode(**kwargs)
        lm.init_output(embeddings[lm.tgt_key].get_vsz(), **kwargs)
        return lm

    def forward(self, input, hidden):
        emb = self.embed(input)
        decoded, hidden = self.decode(emb, hidden)
        return self.output(decoded), hidden

    def init_output(self, vsz, **kwargs):
        unif = float(kwargs.get('unif', 0.0))
        do_weight_tying = bool(kwargs.get('tie_weights', False))
        self.proj = pytorch_linear(self.hsz, vsz, unif)
        if do_weight_tying and self.hsz == self.embeddings[self.tgt_key].get_dsz():
            self.proj.weight = self.embeddings[self.tgt_key].embeddings.weight

    def output(self, x):
        outputs = self.proj(x)
        return outputs


@register_model(task='lm', name='default')
class RNNLanguageModel(LanguageModelBase):

    def __init__(self):
        super(RNNLanguageModel, self).__init__()

    def init_hidden(self, batchsz):
        weight = next(self.parameters()).data
        return (torch.autograd.Variable(weight.new(self.layers, batchsz, self.hsz).zero_()),
                torch.autograd.Variable(weight.new(self.layers, batchsz, self.hsz).zero_()))

    def init_decode(self, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.5))
        vdrop = bool(kwargs.get('variational_dropout', False))
        if vdrop:
            self.rnn_dropout = VariationalDropout(pdrop)
        else:
            self.rnn_dropout = nn.Dropout(pdrop)

        self.rnn = pytorch_lstm(self.dsz, self.hsz, 'lstm', self.layers, pdrop, batch_first=True)

    def decode(self, emb, hidden):
        output, hidden = self.rnn(emb, hidden)
        output = self.rnn_dropout(output).contiguous()
        return output, hidden


@register_model(task='lm', name='transformer')
class TransformerLanguageModel(LanguageModelBase):

    def __init__(self):
        super(TransformerLanguageModel, self).__init__()

    def init_decode(self, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.5))
        layers = kwargs.get('layers', 1)
        d_model = int(kwargs.get('d_model', kwargs.get('hsz')))
        num_heads = kwargs.get('num_heads', 4)
        self.proj_to_dsz = pytorch_linear(self.dsz, d_model)
        self.transformer = TransformerEncoderStack(num_heads, d_model=d_model, pdrop=pdrop, scale=True, layers=layers)

    def decode(self, bth, hidden):
        bth = self.proj_to_dsz(bth)
        T = bth.shape[1]
        mask = subsequent_mask(T).type_as(bth)
        return self.transformer(bth, mask), None
