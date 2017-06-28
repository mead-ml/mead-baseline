import torch
import torch.nn as nn
import math
import json
from baseline.pytorch.torchy import *
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

class ConvModel(nn.Module):

    def save(self, outname):
        print('saving %s' % outname)
        torch.save(self, outname)
        #json.dump(self.vocab, outname + '.vocab')

    def create_loss(self):
        return nn.NLLLoss()

    @staticmethod
    def load(dirname, base):
        name = '%s/%s.model' % (dirname, base)
        model = torch.load(name)
        return model


    @staticmethod
    def load(dirname, base):
        name = '%s/%s.model' % (dirname, base)
        return torch.load(name)


    @staticmethod
    def create(embeddings, labels, **kwargs):
        pdrop = kwargs.get('dropout', 0.5)
        finetune = kwargs.get('finetune', True)
        filtsz = kwargs['filtsz']
        cmotsz = kwargs['cmotsz']
        model = ConvModel(embeddings, labels, filtsz, cmotsz, pdrop, finetune)
        return model

    def __init__(self, embeddings, labels, filtsz, cmotsz, pdrop, finetune):
        super(ConvModel, self).__init__()
        dsz = embeddings.dsz
        self.labels = labels
        nc = len(labels)

        self.vocab = embeddings.vocab
        self.lut = nn.Embedding(embeddings.vsz + 1, dsz)
        del self.lut.weight
        self.lut.weight = nn.Parameter(torch.FloatTensor(embeddings.weights),
                                       requires_grad=finetune)
        self.convs = []
        for i, fsz in enumerate(filtsz):
            pad = fsz//2                
            conv = nn.Sequential(
                nn.Conv1d(dsz, cmotsz, fsz, padding=pad),
                nn.ReLU()
            )
            self.convs.append(conv)
            # Add the module so its managed correctly
            self.add_module('conv-%d' % i, conv)

        # Width of concat of parallel convs
        input_dim = cmotsz * len(filtsz)
        self.fconns = nn.Sequential()

        append2seq(self.fconns, (
            #nn.BatchNorm1d(input_dim),
            nn.Dropout(pdrop),
            nn.Linear(input_dim, nc),
            nn.LogSoftmax()
        ))
    
    def forward(self, input):
        # BxTxC -> BxCxT
        embeddings = self.lut(input).transpose(1, 2).contiguous()
        mots = []
        for conv in self.convs:
            # In Conv1d, data BxCxT, max over time
            conv_out = conv(embeddings)
            mot, _ = conv_out.max(2)
            mots.append(mot.squeeze(2))

        mots = torch.cat(mots, 1)
        output = self.fconns(mots)
        return output


class LSTMModel(nn.Module):

    def save(self, outname):
        print('saving %s' % outname)
        torch.save(self, outname)

    def create_loss(self):
        return nn.NLLLoss()

    @staticmethod
    def load(dirname, base):
        name = '%s/%s.model' % (dirname, base)
        model = torch.load(name)
        return model

    def __init__(self, embeddings, labels, hsz, pdrop, unif, nlayers=1):
        super(LSTMModel, self).__init__()
        dsz = embeddings.dsz
        self.labels = labels
        nc = len(labels)
        
        self.vocab = embeddings.vocab
        self.lut = nn.Embedding(embeddings.vsz + 1, dsz)
        del self.lut.weight

        self.lut.weight = nn.Parameter(torch.FloatTensor(embeddings.weights),
                                       requires_grad=True)

        self.lstm = nn.LSTM(dsz, hsz, nlayers, bias=False, batch_first=True, dropout=pdrop)
        for weight in self.lstm.parameters():
            weight.data.uniform_(-unif, unif)
        # Width of concat of parallel convs
        input_dim = hsz
        self.fconns = nn.Sequential()

        append2seq(self.fconns, (
            nn.BatchNorm1d(input_dim),
            nn.Dropout(pdrop),
            nn.Linear(input_dim, nc),
            nn.LogSoftmax()
        ))
    
    def forward(self, input):
        # BxTxH
        #print(input)
        embeddings = self.lut(input)
        #print(embeddings.size())
        output, hidden = self.lstm(embeddings)
        # This squeeze can cause problems when B=1
        last_frame = output[:,-1,:].squeeze()
        output = self.fconns(last_frame)
        return output

def create_model(w2v, labels, **kwargs):
    #model_type = kwargs.get('model_type', 'conv')
    return ConvModel.create(w2v, labels, **kwargs)
