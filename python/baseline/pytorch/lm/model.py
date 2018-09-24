from baseline.pytorch.torchy import *
from baseline.model import create_lang_model, load_lang_model
import torch.autograd
import math


class AbstractLanguageModel(nn.Module):

    def __init__(self):
        super(AbstractLanguageModel, self).__init__()

    def save(self, outname):
        torch.save(self, outname)

    def create_loss(self):
        return SequenceCriterion(LossFn=nn.CrossEntropyLoss)

    @staticmethod
    def load(outname, **kwargs):
        model = torch.load(outname)
        return model

    def init_hidden(self, batchsz):
        weight = next(self.parameters()).data
        return (torch.autograd.Variable(weight.new(self.nlayers, batchsz, self.hsz).zero_()),
                torch.autograd.Variable(weight.new(self.nlayers, batchsz, self.hsz).zero_()))

    def get_vocab(self, vocab_type='word'):
        return self.word_vocab if vocab_type == 'word' else self.char_vocab

    def _rnnlm(self, emb, hidden):
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output).contiguous()
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

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


class BasicLanguageModel(AbstractLanguageModel):
    def __init__(self):
        super(BasicLanguageModel, self).__init__()

    def get_embeddings_section(self):
        pass

    @classmethod
    def create(cls, embeddings, **kwargs):

        model = cls()
        vectors = embeddings[model.get_embeddings_section()]
        model.gpu = kwargs.get('gpu', True)
        model.hsz = int(kwargs['hsz'])
        unif = float(kwargs.get('unif', 0.0))
        model.nlayers = int(kwargs.get('layers', 1))
        pdrop = float(kwargs.get('dropout', 0.5))
        model.vocab_sz = vectors.vsz

        model.vocab = vectors.vocab
        model.embed = pytorch_embedding(vectors)
        model.dsz = vectors.dsz
        model.dropout = nn.Dropout(pdrop)
        model.rnn, out_hsz = pytorch_lstm(model.dsz, model.hsz, 'lstm', model.nlayers, pdrop, batch_first=True)
        model.decoder = nn.Sequential()
        append2seq(model.decoder, (
            pytorch_linear(model.hsz, model.vocab_sz, unif),
        ))

        return model

    def forward(self, input, hidden):
        emb = self._encoder(input[0])
        return self._rnnlm(emb, hidden)

    def _encoder(self, input):
        return self.dropout(self.embed(input))


class WordLanguageModel(BasicLanguageModel):

    def __init__(self):
        super(WordLanguageModel, self).__init__()

    def get_embeddings_section(self):
        return 'word'


class CharLanguageModel(BasicLanguageModel):

    def __init__(self):
        super(CharLanguageModel, self).__init__()

    def get_embeddings_section(self):
        return 'char'


class CharCompLanguageModel(AbstractLanguageModel):

    def __init__(self):
        super(CharCompLanguageModel, self).__init__()

    def _init_char_encoder(self, char_dsz, char_vec, **kwargs):
        self.cembed = pytorch_embedding(char_vec)
        filtsz = kwargs['cfiltsz']
        cmotsz = kwargs['hsz']
        convs = []
        for i, fsz in enumerate(filtsz):
            pad = fsz//2
            conv = nn.Sequential(
                nn.Conv1d(char_dsz, cmotsz, fsz, padding=pad),
                pytorch_activation("relu")
            )
            convs.append(conv)
            # Add the module so its managed correctly
        self.convs = nn.ModuleList(convs)

        wchsz = cmotsz * len(filtsz)
        self.highway = nn.Sequential()
        append2seq(self.highway, (
            Highway(wchsz),
            Highway(wchsz)
        ))

        # Width of concat of parallel convs
        return wchsz

    def _char_encoder(self, batch_first_words):
        emb = self.dropout(self.cembed(batch_first_words))
        embeddings = emb.transpose(1, 2).contiguous()
        mots = []
        for conv in self.convs:
            # In Conv1d, data BxCxT, max over time
            conv_out = conv(embeddings)
            mot, _ = conv_out.max(2)
            mots.append(mot)

        mots = torch.cat(mots, 1)
        output = self.highway(mots)
        return self.dropout(output)


    @staticmethod
    def create(embeddings, **kwargs):
        word_vec = embeddings.get('word', None)

        model = CharCompLanguageModel()

        model.gpu = kwargs.get('gpu', True)
        char_dsz = kwargs.get('charsz', 0)
        cmotsz_all = model._init_char_encoder(char_dsz, embeddings['char'], **kwargs)
        word_dsz = 0
        model.hsz = int(kwargs['hsz'])
        unif = float(kwargs.get('unif', 0.0))
        model.nlayers = int(kwargs.get('layers', 1))
        pdrop = float(kwargs.get('dropout', 0.5))
        model.vocab_sz = word_vec.vsz

        #if word_vec is not None:
        #    model.word_vocab = word_vec.vocab
        #    model.wembed = pytorch_embedding(word_vec)
        #    word_dsz = word_vec.dsz

        model.dropout = nn.Dropout(pdrop)
        model.rnn, out_hsz = pytorch_lstm(cmotsz_all, model.hsz, 'lstm', model.nlayers, pdrop, batch_first=True)
        model.decoder = nn.Sequential()
        append2seq(model.decoder, (
            pytorch_linear(out_hsz, model.vocab_sz, unif),
        ))

        return model

    def make_input(self, batch_dict):
        x, y = super(CharCompLanguageModel, self).make_input(batch_dict)
        xch = torch.from_numpy(batch_dict['xch'])

        if self.gpu:
            xch = xch.cuda()

        return x, torch.autograd.Variable(xch), y

    def forward(self, input, hidden):
        ##x = input[0]
        xch = input[1]
        # BTC
        bt_x_w = xch.view(xch.size(0) * xch.size(1), -1)
        b_t_wch = self._char_encoder(bt_x_w).view(xch.size(0), xch.size(1), -1)

        #if self.wembed is not None:
        #    emb = self.dropout(self.wembed(x))
        #    emb = torch.cat([emb, b_t_wch])
        #else:
        emb = b_t_wch
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
    'default': WordLanguageModel.create,
    'char': CharLanguageModel.create,
    'convchar': CharCompLanguageModel.create
}

BASELINE_LM_LOADERS = {
    'default': WordLanguageModel.load,
    'char': CharLanguageModel.load,
    'convchar': CharCompLanguageModel.load
}

def create_model(embeddings, **kwargs):
    lm = create_lang_model(BASELINE_LM_MODELS, embeddings, **kwargs)
    return lm

def load_model(modelname, **kwargs):
    return load_lang_model(BASELINE_LM_LOADERS, modelname, **kwargs)
