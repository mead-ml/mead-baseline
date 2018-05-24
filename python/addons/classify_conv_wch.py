from baseline.pytorch.classify import *


cudnn.benchmark = True


class ConvConvModel(nn.Module, Classifier):

    def __init__(self):
        super(ConvConvModel, self).__init__()

    @classmethod
    def load(cls, outname, **kwargs):
        model = torch.load(outname)
        return model

    def save(self, outname):
        print('saving %s' % outname)
        torch.save(self, outname)

    @classmethod
    def create(cls, embeddings_set, labels, **kwargs):

        word_embeddings = embeddings_set['word']
        char_embeddings = embeddings_set['char']
        word_filtsz = kwargs.get('filtsz', [3, 4, 5])
        char_filtsz = kwargs.get('cfiltsz', [3])
        word_hsz = kwargs.get('cmotsz', kwargs.get('hsz', 100))
        char_hsz = kwargs.get('char_hsz', 30)
        nc = len(labels)
        finetune = kwargs.get('finetune', True)

        model = cls()
        model.gpu = not bool(kwargs.get('nogpu', False))

        model.word_dsz = word_embeddings.dsz
        model.char_dsz = char_embeddings.dsz
        model.pdrop = kwargs.get('dropout', 0.5)
        model.labels = labels

        model.lut = pytorch_embedding(word_embeddings, finetune)
        model.vocab = word_embeddings.vocab
        model.char_lut = pytorch_embedding(char_embeddings)
        model.char_vocab = char_embeddings.vocab
        activation_type = kwargs.get('activation', 'relu')
        model.char_comp = ParallelConv(model.char_dsz, char_hsz, char_filtsz, activation_type, model.pdrop)
        input_sz = model.word_dsz + model.char_comp.outsz
        model.word_comp = ParallelConv(input_sz, word_hsz, word_filtsz, activation_type, model.pdrop)
        model.proj = nn.Linear(model.word_comp.outsz, nc)
        model.log_softmax = nn.LogSoftmax(dim=1)
        print(model)
        return model

    def create_loss(self):
        return nn.NLLLoss()

    def make_input(self, batch_dict):
        x = batch_dict['x']
        xch = batch_dict['xch']
        y = batch_dict.get('y')
        if self.gpu:
            x = x.cuda()
            xch = xch.cuda()
            if y is not None:
                y = y.cuda()

        return x, xch, y

    def _char_encoding(self, xch):

        # For starters we need to perform embeddings for each character
        # (TxB) x W -> (TxB) x W x D
        char_embeds = self.char_lut(xch)
        # (TxB) x D x W
        char_vecs = char_embeds.transpose(1, 2).contiguous()
        mots = self.char_comp(char_vecs)
        ##output = self.word_ch_embed(mots)
        return mots ##+ output

    def forward(self, input):
        # BxTxC
        x = input[0]
        xch = input[1]
        B, T, W = xch.shape
        embeddings_word = self.lut(x)
        embeddings_char = self._char_encoding(xch.view(-1, W)).view(B, T, self.char_comp.outsz)
        embeddings = torch.cat([embeddings_word, embeddings_char], 2)
        embeddings = embeddings.transpose(1, 2).contiguous()
        pooled = self.word_comp(embeddings)
        return self.log_softmax(self.proj(pooled))

    def classify(self, batch_dict):
        x = batch_dict['x']
        xch = batch_dict['xch']
        with torch.no_grad():
            if self.gpu:
                x = x.cuda()
                xch = xch.cuda()
            probs = model(x, xch).cuda().exp()
            probs.div_(torch.sum(probs))
            results = []
            batchsz = probs.size(0)
            for b in range(batchsz):
                outcomes = [(model.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
                results.append(outcomes)
        return results

    def get_labels(self):
        return self.labels

    def get_vocab(self):
        return self.vocab


def create_model(embeddings, labels, **kwargs):
    return ConvConvModel.create(embeddings, labels, **kwargs)


def load_model(modelname, **kwargs):
    return ConvConvModel.load(modelname, **kwargs)

