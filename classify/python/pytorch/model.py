import torch
import torch.nn as nn

def _append2seq(seq, modules):
    for module in modules:
        seq.add_module(str(module), module)

class ConvModel(nn.Module):

    def save(self, outdir, base):
        outname = '%s/%s.model' % (outdir, base)
        torch.save(self, outname)

    def create_loss(self):
        return nn.NLLLoss()


    @staticmethod
    def load(dirname, base):
        name = '%s/%s.model' % (dirname, base)
        return torch.load(name)

    def __init__(self, embeddings, nc, filtsz, cmotsz, hsz, pdrop, finetune):
        super(ConvModel, self).__init__()
        dsz = embeddings.dsz
        self.lut = nn.Embedding(embeddings.vsz + 1, dsz)
        del self.lut.weight
        self.lut.weight = nn.Parameter(torch.FloatTensor(embeddings.weights),
                                       requires_grad=finetune)
        self.convs = []
        for fsz in filtsz:
            conv = nn.Sequential(
                nn.Conv1d(dsz, cmotsz, fsz),
                nn.ReLU()
            )
            self.convs.append(conv)
            # Add the module so its managed correctly
            self.add_module('conv-%d' % fsz, conv)

        # Width of concat of parallel convs
        input_dim = cmotsz * len(filtsz)
        self.fconns = nn.Sequential()

        # Using additional hidden layer?
        if hsz > 0:
            _append2seq(self.fconns, ( 
                nn.Dropout(pdrop),
                nn.Linear(input_dim, hsz),
                nn.ReLU()
            ))
            input_dim = hsz
        _append2seq(self.fconns, (
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
            mot, _ = conv(embeddings).max(2)
            mots.append(mot.squeeze(2))

        mots = torch.cat(mots, 1)
        output = self.fconns(mots)
        return output
