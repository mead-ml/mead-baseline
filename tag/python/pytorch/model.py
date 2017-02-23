import torch
import torch.nn as nn


class SequenceCriterion(nn.Module):

    def __init__(self, nc):
        super(SequenceCriterion, self).__init__()
        # Assume pad is zero element for now
        weight = torch.ones(nc)
        weight[0] = 0
        self.crit = nn.NLLLoss(weight, size_average=False)
    
    def forward(self, inputs, targets):
        # This is BxT, which is what we want!
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return loss


def _conv1d(in_channels, out_channels, fsz, unif):
    c = nn.Conv1d(in_channels, out_channels, fsz)
    c.weight.data.uniform_(-unif, unif)
    return c


def _linear(in_sz, out_sz, unif):
    l = nn.Linear(in_sz, out_sz)
    l.weight.data.uniform_(-unif, unif)
    return l


def _rnn(insz, hsz, rnntype, nlayers, unif):
    # For now, just handle LSTM, should be easy to update later
    ndir = 2 if rnntype.startswith('b') else 1
    rnn = torch.nn.LSTM(insz, hsz, nlayers, bidirectional=True if ndir > 1 else False, bias=False)
    for weight in rnn.parameters():
        weight.data.uniform_(-unif, unif)
    return rnn, ndir*hsz


def _embedding(x2vec, finetune=True):
    dsz = x2vec.dsz
    lut = nn.Embedding(x2vec.vsz + 1, dsz, padding_idx=0)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(x2vec.weights),
                              requires_grad=finetune)
    return lut


def _append2seq(seq, modules):
    for module in modules:
        seq.add_module(str(module), module)


class TaggerModel(nn.Module):

    def save(self, outdir, base):
        outname = '%s/%s.model' % (outdir, base)
        torch.save(self, outname)

    def create_loss(self):
        return SequenceCriterion(self.nc)

    @staticmethod
    def load(dirname, base):
        name = '%s/%s.model' % (dirname, base)
        return torch.load(name)

    def _char_word_conv_embeddings(self, maxw, filtsz, char_dsz, wchsz, pdrop, unif):
        self.char_convs = []
        for fsz in filtsz:
            conv = nn.Sequential(
                _conv1d(char_dsz, wchsz, fsz, unif),
                nn.ReLU()
            )
            self.char_convs.append(conv)
            # Add the module so its managed correctly
            self.add_module('char-conv-%d' % fsz, conv)

        # Width of concat of parallel convs
        self.wchsz = wchsz * len(filtsz)
        self.word_ch_embed = nn.Sequential()
        _append2seq(self.word_ch_embed, (
            nn.Dropout(pdrop),
            _linear(self.wchsz, self.wchsz, unif),
            nn.ReLU()
        ))

    def __init__(self, labels, word_vec, char_vec, mxlen, maxw, rnntype, wchsz, hsz, filtsz, pdrop, unif, nlayers=1):
        super(TaggerModel, self).__init__()
        char_dsz = char_vec.dsz
        word_dsz = 0
        self.nc = len(labels)

        self._char_word_conv_embeddings(maxw, filtsz, char_dsz, wchsz, pdrop, unif)

        if word_vec is not None:
            self.wembed = _embedding(word_vec)
            word_dsz = word_vec.dsz

        self.cembed = _embedding(char_vec)
        self.dropout = nn.Dropout(pdrop)
        self.rnn, hsz = _rnn(self.wchsz + word_dsz, hsz, rnntype, nlayers, unif)
        self.decoder = _linear(hsz, self.nc, unif)
        self.softmax = nn.LogSoftmax()

    def char2word(self, xch_i):
        # For starters we need to perform embeddings for each character
        char_vecs = self.cembed(xch_i).transpose(1, 2).contiguous()
        mots = []
        for conv in self.char_convs:
            # In Conv1d, data BxCxT, max over time
            mot, _ = conv(char_vecs).max(2)
            mots.append(mot.squeeze(2))

        mots = torch.cat(mots, 1)
        output = self.word_ch_embed(mots)
        return output + mots

    # Input better be xch, x
    def forward(self, input):

        # Temporarily do nothing with xch...
        xch = input[0].transpose(0, 1).contiguous()

        batchsz = xch.size(1)
        seqlen = xch.size(0)
        x = input[1].transpose(0, 1).contiguous() if len(input) == 2 else None
        
        # Vectorized
        words_over_time = self.char2word(xch.view(seqlen * batchsz, -1)).view(seqlen, batchsz, -1)
                
        if x is not None:
            word_vectors = self.wembed(x)
            words_over_time = torch.cat([words_over_time, word_vectors], 2)

        dropped = self.dropout(words_over_time)
        output, hidden = self.rnn(dropped)
        
        # Reform batch as (T x B, D)
        decoded = self.softmax(self.decoder(output.view(output.size(0)*output.size(1),
                                                        -1)))
        # back to T x B x H -> B x T x H
        decoded = decoded.view(output.size(0), output.size(1), -1)
        return decoded.transpose(0, 1).contiguous()
