from baseline.pytorch.torchy import *
from baseline.model import Tagger, create_tagger_model, load_tagger_model
import torch.autograd
import math


class RNNTaggerModel(nn.Module, Tagger):

    def save(self, outname):
        torch.save(self, outname)

    def to_gpu(self):
        self.gpu = True
        self.cuda()
        self.crit.cuda()
        return self

    @staticmethod
    def load(outname, **kwargs):
        model = torch.load(outname)
        return model

    def _char_word_conv_embeddings(self, filtsz, char_dsz, wchsz, pdrop):
        self.char_comp = ParallelConv(char_dsz, wchsz, filtsz, self.activation_type, pdrop)
        # Width of concat of parallel convs
        self.wchsz = wchsz * len(filtsz)
        self.word_ch_embed = nn.Sequential()
        append2seq(self.word_ch_embed, (
            #nn.Dropout(pdrop),
            pytorch_linear(self.wchsz, self.wchsz),
            pytorch_activation(self.activation_type)
        ))

    def __init__(self):
        super(RNNTaggerModel, self).__init__()

    @staticmethod
    def create(labels, embeddings, **kwargs):
        model = RNNTaggerModel()
        word_vec = embeddings['word']
        char_vec = embeddings['char']
        char_dsz = char_vec.dsz
        word_dsz = 0
        hsz = int(kwargs['hsz'])
        model.proj = bool(kwargs.get('proj', False))
        model.use_crf = bool(kwargs.get('crf', False))
        model.crf_mask = bool(kwargs.get('crf_mask', False))
        model.span_type = kwargs.get('span_type')
        model.activation_type = kwargs.get('activation', 'tanh')
        nlayers = int(kwargs.get('layers', 1))
        rnntype = kwargs.get('rnntype', 'blstm')
        model.gpu = False
        print('RNN [%s]' % rnntype)
        wsz = kwargs.get('wsz', 30)
        filtsz = kwargs.get('cfiltsz')

        pdrop = float(kwargs.get('dropout', 0.5))
        model.pdropin_value = float(kwargs.get('dropin', 0.0))
        model.labels = labels
        model._char_word_conv_embeddings(filtsz, char_dsz, wsz, pdrop)

        if word_vec is not None:
            model.word_vocab = word_vec.vocab
            model.wembed = pytorch_embedding(word_vec)
            word_dsz = word_vec.dsz

        model.char_vocab = char_vec.vocab
        model.cembed = pytorch_embedding(char_vec)
        model.dropout = nn.Dropout(pdrop)
        model.rnn, out_hsz = pytorch_lstm(model.wchsz + word_dsz, hsz, rnntype, nlayers, pdrop)
        model.decoder = nn.Sequential()
        if model.proj is True:
            append2seq(model.decoder, (
                pytorch_linear(out_hsz, hsz),
                pytorch_activation(model.activation_type),
                nn.Dropout(pdrop),
                pytorch_linear(hsz, len(model.labels))
            ))
        else:
            append2seq(model.decoder, (
                pytorch_linear(out_hsz, len(model.labels)),
            ))

        if model.use_crf:
            if model.crf_mask:
                assert model.span_type is not None, "A crf mask cannot be used without providing `span_type`"
                model.crf = CRF(
                    len(labels),
                    (model.labels.get("<GO>"), model.labels.get("<EOS>")), batch_first=False,
                    vocab=model.labels, span_type=model.span_type, pad_idx=model.labels.get("<PAD>")
                )
            else:
                model.crf = CRF(len(labels), (model.labels.get("<GO>"), model.labels.get("<EOS>")), batch_first=False)
        model.crit = SequenceCriterion(LossFn=nn.CrossEntropyLoss)
        print(model)
        return model

    def char2word(self, xch_i):

        # For starters we need to perform embeddings for each character
        # (TxB) x W -> (TxB) x W x D
        char_embeds = self.cembed(xch_i)
        # (TxB) x D x W
        char_vecs = char_embeds.transpose(1, 2).contiguous()
        mots = self.char_comp(char_vecs)
        output = self.word_ch_embed(mots)
        return mots + output

    def make_input(self, batch_dict):

        x = batch_dict['x']
        xch = batch_dict['xch']
        y = batch_dict.get('y', None)
        lengths = batch_dict['lengths']
        ids = batch_dict.get('ids', None)

        if self.training and self.pdropin_value > 0.0:
            UNK = self.word_vocab['<UNK>']
            PAD = self.word_vocab['<PAD>']
            mask_pad = x != PAD
            mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(self.pdropin_value).byte()
            x.masked_fill_(mask_pad & mask_drop, UNK)

        lengths, perm_idx = lengths.sort(0, descending=True)
        x = x[perm_idx]
        xch = xch[perm_idx]
        if y is not None:
            y = y[perm_idx]

        if ids is not None:
            ids = ids[perm_idx]

        if self.gpu:
            x = x.cuda()
            xch = xch.cuda()
            if y is not None:
                y = y.cuda()

        if y is not None:
            y = torch.autograd.Variable(y.contiguous())

        return torch.autograd.Variable(x), torch.autograd.Variable(xch), lengths, y, ids

    def _compute_unary_tb(self, x, xch, lengths):

        batchsz = xch.size(1)
        seqlen = xch.size(0)

        words_over_time = self.char2word(xch.view(seqlen * batchsz, -1)).view(seqlen, batchsz, -1)

        if x is not None:
            #print(self.wembed.weight[0])
            word_vectors = self.wembed(x)
            words_over_time = torch.cat([words_over_time, word_vectors], 2)

        dropped = self.dropout(words_over_time)
        # output = (T, B, H)

        packed = torch.nn.utils.rnn.pack_padded_sequence(dropped, lengths.tolist())
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

        # stack (T x B, H)
        #output = self.dropout(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), -1))

        # back to T x B x H
        decoded = decoded.view(output.size(0), output.size(1), -1)

        return decoded

    # Input better be xch, x
    def forward(self, input):
        x = input[0].transpose(0, 1).contiguous()
        xch = input[1].transpose(0, 1).contiguous()
        lengths = input[2]
        batchsz = xch.size(1)
        seqlen = xch.size(0)

        probv = self._compute_unary_tb(x, xch, lengths)
        if self.use_crf is True:
            lengths = lengths.cuda()
            preds, _ = self.crf.decode(probv, lengths)
        else:
            # Get batch (B, T)
            probv = probv.transpose(0, 1)
            preds = []
            for pij, sl in zip(probv, lengths):
                _, unary = torch.max(pij[:sl], 1)
                preds.append(unary.data)

        return preds

    def compute_loss(self, input):
        x = input[0].transpose(0, 1).contiguous()
        xch = input[1].transpose(0, 1).contiguous()
        lengths = input[2]
        tags = input[3]

        probv = self._compute_unary_tb(x, xch, lengths)
        batch_loss = 0.
        total_tags = 0.
        if self.use_crf is True:
            # Get tags as [T, B]
            tags = tags.transpose(0, 1)
            lengths = lengths.cuda()
            batch_loss = torch.mean(self.crf.neg_log_loss(probv, tags.data, lengths))
        else:
            # Get batch (B, T)
            probv = probv.transpose(0, 1)
            for pij, gold, sl in zip(probv, tags, lengths):
                unary = pij[:sl]
                gold_tags = gold[:sl]
                total_tags += len(gold_tags)
                batch_loss += self.crit(unary, gold_tags)
            batch_loss /= probv.shape[0]

        return batch_loss

    def get_vocab(self, vocab_type='word'):
        return self.word_vocab if vocab_type == 'word' else self.char_vocab

    def get_labels(self):
        return self.labels

    def predict(self, batch_dict):
        x = batch_dict['x']
        xch = batch_dict['xch']
        lengths = batch_dict['lengths']
        return predict_seq_bt(self, x, xch, lengths)

BASELINE_TAGGER_MODELS = {
    'default': RNNTaggerModel.create,
}

BASELINE_TAGGER_LOADERS = {
    'default': RNNTaggerModel.load,
}

def create_model(labels, embeddings, **kwargs):
    return create_tagger_model(BASELINE_TAGGER_MODELS, labels, embeddings, **kwargs)


def load_model(modelname, **kwargs):
    return load_tagger_model(BASELINE_TAGGER_LOADERS, modelname, **kwargs)
