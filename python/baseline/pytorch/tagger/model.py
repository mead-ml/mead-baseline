from baseline.pytorch.torchy import *
from baseline.model import Tagger


class TaggerModel(nn.Module, Tagger):

    def save(self, outname):
        torch.save(self, outname)

    def create_loss(self):
        return SequenceCriterion(len(self.labels))

    @staticmethod
    def load(outname, **kwargs):
        model = torch.load(outname)
        return model

    def _char_word_conv_embeddings(self, filtsz, char_dsz, wchsz, pdrop, unif):
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

    def __init__(self):
        super(TaggerModel, self).__init__()

    @staticmethod
    def create(labels, word_vec, char_vec, **kwargs):
        model = TaggerModel()
        char_dsz = char_vec.dsz
        word_dsz = 0
        hsz = int(kwargs['hsz'])
        nlayers = int(kwargs.get('layers', 1))
        rnntype = kwargs.get('rnntype', 'lstm')
        print('RNN [%s]' % rnntype)
        unif = float(kwargs.get('unif', 0.25))
        wsz = kwargs.get('wsz', 30)
        filtsz = kwargs.get('cfiltsz')
        crf = bool(kwargs.get('crf', False))
        if crf:
            print('Warning: CRF not supported yet in PyTorch model... ignoring')
        pdrop = float(kwargs.get('dropout', 0.5))
        model.labels = labels
        model._char_word_conv_embeddings(filtsz, char_dsz, wsz, pdrop, unif)

        if word_vec is not None:
            model.word_vocab = word_vec.vocab
            model.wembed = pytorch_embedding(word_vec)
            word_dsz = word_vec.dsz

        model.char_vocab = char_vec.vocab
        model.cembed = pytorch_embedding(char_vec)
        model.dropout = nn.Dropout(pdrop)
        model.rnn, hsz = pytorch_lstm(model.wchsz + word_dsz, hsz, rnntype, nlayers, pdrop, unif)
        model.decoder = pytorch_linear(hsz, len(model.labels), unif)
        model.softmax = nn.LogSoftmax()
        return model

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

    def get_vocab(self, vocab_type='word'):
        return self.word_vocab if vocab_type == 'word' else self.char_vocab

    def get_labels(self):
        return self.labels

    def predict(self, x, xch, lengths):
        return predict_seq_bt(self, x, xch, lengths)


def create_model(labels, word_embedding, char_embedding, **kwargs):
    model = TaggerModel.create(labels, word_embedding, char_embedding, **kwargs)
    return model
