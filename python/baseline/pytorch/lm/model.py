from baseline.pytorch.torchy import *
from baseline.model import create_lang_model, load_lang_model, LanguageModel
import torch.autograd
import math


class BasicLanguageModel(nn.Module, LanguageModel):
    def __init__(self):
        super(BasicLanguageModel, self).__init__()

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
        return (torch.autograd.Variable(weight.new(self.layers, batchsz, self.hsz).zero_()),
                torch.autograd.Variable(weight.new(self.layers, batchsz, self.hsz).zero_()))

    def _rnnlm(self, emb, hidden):
        output, hidden = self.rnn(emb, hidden)
        output = self.rnn_dropout(output).contiguous()
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

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

    def _embed(self, input):
        all_embeddings = []
        for k, embedding in self.embeddings.items():
            all_embeddings += [embedding.encode(input[k])]
        return torch.cat(all_embeddings, 2)

    def _init_embed(self, embeddings, **kwargs):
        self.embeddings = EmbeddingsContainer()
        input_sz = 0
        for k, embedding in embeddings.items():
            self.embeddings[k] = embedding
            input_sz += embedding.get_dsz()
        return input_sz

    @classmethod
    def create(cls, embeddings, **kwargs):

        lm = cls()
        lm.gpu = kwargs.get('gpu', True)
        lm.hsz = int(kwargs['hsz'])
        unif = float(kwargs.get('unif', 0.0))
        lm.layers = int(kwargs.get('layers', 1))
        pdrop = float(kwargs.get('dropout', 0.5))
        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination ')

        lm.dsz = lm._init_embed(embeddings, **kwargs)
        lm.dropout = nn.Dropout(pdrop)
        lm.rnn, out_hsz = pytorch_lstm(lm.dsz, lm.hsz, 'lstm', lm.layers, pdrop, batch_first=True)
        lm.decoder = nn.Sequential()
        append2seq(lm.decoder, (
            pytorch_linear(lm.hsz, embeddings[lm.tgt_key].get_vsz(), unif),
        ))
        lm.vdrop = bool(kwargs.get('variational_dropout', False))

        if lm.vdrop:
            lm.rnn_dropout = VariationalDropout(pdrop)
        else:
            lm.rnn_dropout = nn.Dropout(pdrop)

        return lm

    def forward(self, input, hidden):
        emb = self._encoder(input)
        return self._rnnlm(emb, hidden)

    def _encoder(self, input):
        return self.dropout(self._embed(input))
"""

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
"""

BASELINE_LM_MODELS = {
    'default': BasicLanguageModel.create
}

BASELINE_LM_LOADERS = {
    'default': BasicLanguageModel.load
}

def create_model(embeddings, **kwargs):
    lm = create_lang_model(BASELINE_LM_MODELS, embeddings, **kwargs)
    return lm

def load_model(modelname, **kwargs):
    return load_lang_model(BASELINE_LM_LOADERS, modelname, **kwargs)
