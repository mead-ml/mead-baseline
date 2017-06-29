from baseline.pytorch.torchy import *

class TaggerModel(nn.Module):

    def save(self, outname):
        torch.save(self, outname)

    def create_loss(self):
        return SequenceCriterion(len(self.labels))

    @staticmethod
    def load(dirname, base):
        name = '%s/%s.model' % (dirname, base)
        return torch.load(name)

    def _char_word_conv_embeddings(self, maxw, filtsz, char_dsz, wchsz, pdrop, unif):
        self.char_convs = []
        for fsz in filtsz:
            conv = nn.Sequential(
                pytorch_conv1d(char_dsz, wchsz, fsz, unif),
                nn.ReLU()
            )
            self.char_convs.append(conv)
            # Add the module so its managed correctly
            self.add_module('char-conv-%d' % fsz, conv)

        # Width of concat of parallel convs
        self.wchsz = wchsz * len(filtsz)
        self.word_ch_embed = nn.Sequential()
        append2seq(self.word_ch_embed, (
            nn.Dropout(pdrop),
            pytorch_linear(self.wchsz, self.wchsz, unif),
            nn.ReLU()
        ))

    def __init__(self, labels, word_vec, char_vec, mxlen, maxw, rnntype, wchsz, hsz, filtsz, pdrop, unif, nlayers=1):
        super(TaggerModel, self).__init__()
        char_dsz = char_vec.dsz
        word_dsz = 0
        self.labels = labels
        self._char_word_conv_embeddings(maxw, filtsz, char_dsz, wchsz, pdrop, unif)

        if word_vec is not None:
            self.wembed = pytorch_embedding(word_vec)
            word_dsz = word_vec.dsz

        self.cembed = pytorch_embedding(char_vec)
        self.dropout = nn.Dropout(pdrop)
        self.rnn, hsz = pytorch_lstm(self.wchsz + word_dsz, hsz, rnntype, nlayers, pdrop, unif)
        self.decoder = pytorch_linear(hsz, len(self.labels), unif)
        self.softmax = nn.LogSoftmax()

    def char2word(self, xch_i):
        # For starters we need to perform embeddings for each character
        char_embeds = self.cembed(xch_i)
        char_vecs = char_embeds.transpose(1, 2).contiguous()
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

        xch = input[1].transpose(0, 1).contiguous()
        batchsz = xch.size(1)
        seqlen = xch.size(0)
        x = input[0].transpose(0, 1).contiguous()
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


def create_model(labels, word_embedding, char_embedding, **kwargs):
    hsz = int(kwargs['hsz'])
    layers = int(kwargs.get('layers', 1))
    rnntype = kwargs.get('rnntype', 'lstm')
    maxs = kwargs.get('maxs', 100)
    maxw = kwargs.get('maxw', 100)
    unif = float(kwargs.get('unif', 0.25))
    wsz = kwargs.get('wsz', 30)
    filtsz = kwargs.get('cfiltsz')
    crf = bool(kwargs.get('crf', False))
    if crf:
        print('Warning: CRF not supported yet in PyTorch model... ignoring')
    dropout = float(kwargs.get('dropout', 0.5))
    model = TaggerModel(labels, word_embedding, char_embedding, maxs, maxw,
                        rnntype, wsz, hsz, filtsz, dropout, unif, layers)
    return model
