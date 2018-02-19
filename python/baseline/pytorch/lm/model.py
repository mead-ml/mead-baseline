from baseline.pytorch.torchy import *
from baseline.model import create_lang_model  #, load_lang_model
import torch.autograd
import math


class WordLanguageModel(nn.Module):

    def save(self, outname):
        torch.save(self, outname)

    def create_loss(self):
        return SequenceCriterion(LossFn=nn.CrossEntropyLoss)

    @staticmethod
    def load(outname, **kwargs):
        model = torch.load(outname)
        return model

    def __init__(self):
        super(WordLanguageModel, self).__init__()

    @staticmethod
    def create(embeddings, **kwargs):
        word_vec = embeddings['word']

        model = WordLanguageModel()
        model.gpu = kwargs.get('gpu', True)
        word_dsz = 0
        model.hsz = int(kwargs['hsz'])
        unif = float(kwargs.get('unif', 0.0))
        print(unif)
        model.nlayers = int(kwargs.get('layers', 1))
        pdrop = float(kwargs.get('dropout', 0.5))
        model.vocab_sz = word_vec.vsz + 1

        if word_vec is not None:
            model.word_vocab = word_vec.vocab
            model.wembed = pytorch_embedding(word_vec)
            word_dsz = word_vec.dsz

        model.dropout = nn.Dropout(pdrop)
        model.rnn, out_hsz = pytorch_lstm(word_dsz, model.hsz, 'lstm', model.nlayers, pdrop, batch_first=True)
        model.decoder = nn.Sequential()
        append2seq(model.decoder, (
                pytorch_linear(model.hsz, model.vocab_sz, unif),
            ))

        return model

    def make_input(self, batch_dict):

        x = torch.from_numpy(batch_dict['x'])
        y = batch_dict.get('y', None)

        if y is not None:
            y = torch.from_numpy(y)

        if self.gpu:
            x = x.cuda()
            if y is not None:
                y = y.cuda()

        return torch.autograd.Variable(x), torch.autograd.Variable(y) if y is not None else None

    def forward(self, input, hidden):
        emb = self.dropout(self.wembed(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output).contiguous()
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batchsz):
        weight = next(self.parameters()).data
        return (torch.autograd.Variable(weight.new(self.nlayers, batchsz, self.hsz).zero_()),
                torch.autograd.Variable(weight.new(self.nlayers, batchsz, self.hsz).zero_()))

    def get_vocab(self, vocab_type='word'):
        return self.word_vocab if vocab_type == 'word' else self.char_vocab


BASELINE_LM_MODELS = {
    'default': WordLanguageModel.create
}


def create_model(embeddings, **kwargs):
    lm = create_lang_model(BASELINE_LM_MODELS, embeddings, **kwargs)
    return lm


#def load_model(modelname, **kwargs):
#    return load_lang_model(WordLanguageModel.load, modelname, **kwargs)
