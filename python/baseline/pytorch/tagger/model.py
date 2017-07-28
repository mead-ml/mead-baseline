from baseline.pytorch.torchy import *
from baseline.model import Tagger
import torch.autograd

# Some of this code is borrowed from here:
# https://github.com/rguthrie3/DeepLearningForNLPInPytorch
# I have vectorized the implementation for reasonable performance on real data
# Helper functions to make the code more readable.
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# This code is not used, its here only for reference!
def forward_algorithm(unary, transitions, start_idx, end_idx):
    siglen, num_labels = unary.size()

    # Do the forward algorithm to compute the partition function
    init_alphas = torch.Tensor(1, num_labels).fill_(-10000.).cuda()
    # START_TAG has all of the score.
    init_alphas[0][start_idx] = 0.

    # Wrap in a variable so that we will get automatic backprop
    alphas = torch.autograd.Variable(init_alphas)

    # Iterate through the sentence
    for unary_t in unary:
        # torch.Size([T, num_labels])
        #print(unary_t.size())
        alphas_t = []  # The forward variables at this timestep
        for tag in range(num_labels):
            # broadcast the emission score: it is the same regardless of the previous tag
            #            [1x1]                           [1 x L] (replicated)
            emit_score = unary_t[tag].view(1, -1).expand(1, num_labels)
            # the ith entry of trans_score is the score of transitioning to tag from i
            #             [1xL]
            trans_score = transitions[tag].view(1, -1)
            # The ith entry of next_tag_var is the value for the edge (i -> tag)
            # before we do log-sum-exp
            #              [1xL]
            next_tag_var = alphas + trans_score + emit_score
            # The forward variable for this tag is log-sum-exp of all the scores.
            alphas_t.append(log_sum_exp(next_tag_var))
        alphas = torch.cat(alphas_t).view(1, -1)
    terminal_var = alphas + transitions[end_idx]
    alpha = log_sum_exp(terminal_var)
    return alpha


def vec_log_sum_exp(vec, dim=2):
    """Vectorized version of log-sum-exp
    
    :param vec: Vector
    :param dim: What dimension to operate on
    :return: 
    """
    max_scores, idx = torch.max(vec, dim)
    max_scores_broadcast = max_scores.expand_as(vec)
    return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores_broadcast), dim))


def forward_algorithm_vec(unary, transitions, start_idx, end_idx):
    """Vectorized forward algorithm for CRF layer
    
    :param unary: The observations
    :param transitions: The transitions
    :param start_idx: The index of the start position
    :param end_idx: The index of the end position
    :return: Alphas
    """
    siglen, num_labels = unary.size()

    # Do the forward algorithm to compute the partition function
    init_alphas = torch.Tensor(1, num_labels).fill_(-10000.).cuda()
    # START_TAG has all of the score.
    init_alphas[0][start_idx] = 0.

    # Wrap in a variable so that we will get automatic backprop
    alphas = torch.autograd.Variable(init_alphas)

    # Iterate through the sentence
    for t, unary_t in enumerate(unary):
        expanded_alpha_t = alphas.expand(num_labels, num_labels)
        # torch.Size([T, num_labels])
        emit_scores_transpose = torch.autograd.Variable(unary_t.data.view(1, -1).transpose(0, 1).expand(num_labels, num_labels), requires_grad=False)
        next_tag_var = expanded_alpha_t + transitions
        next_tag_var += emit_scores_transpose
        scores = vec_log_sum_exp(next_tag_var, 1).transpose(0, 1)
        alphas = scores

    terminal_var = alphas + transitions[end_idx]
    alpha = log_sum_exp(terminal_var)
    return alpha


def viterbi_decode(unary, transitions, start_idx, end_idx):
    backpointers = []

    siglen, num_labels = unary.size()
    inits = torch.Tensor(1, num_labels).fill_(-10000.).cuda()
    inits[0][start_idx] = 0

    # alphas at step i holds the viterbi variables for step i-1
    alphas = torch.autograd.Variable(inits)
    for unary_t in unary:
        backpointers_t = []  # holds the backpointers for this step
        viterbi_t = []  # holds the viterbi variables for this step

        for tag in range(num_labels):
            next_tag_var = alphas + transitions[tag]
            best_tag_id = argmax(next_tag_var)
            backpointers_t.append(best_tag_id)
            viterbi_t.append(next_tag_var[0][best_tag_id])
        alphas = (torch.cat(viterbi_t) + unary_t).view(1, -1)
        backpointers.append(backpointers_t)

    # Transition to STOP_TAG
    terminal_var = alphas + transitions[end_idx]
    best_tag_id = argmax(terminal_var)
    path_score = terminal_var[0][best_tag_id]

    # Follow the back pointers to decode the best path.
    best_path = [best_tag_id]
    for backpointers_t in reversed(backpointers):
        best_tag_id = backpointers_t[best_tag_id]
        best_path.append(best_tag_id)
    # Pop off the start tag (we dont want to return that to the caller)
    start = best_path.pop()
    assert start == start_idx
    best_path.reverse()
    return torch.LongTensor(best_path), path_score


def score_sentence(unary, tags, transitions, start_idx, end_idx):
    # Gives the score of a provided tag sequence
    score = torch.autograd.Variable(torch.Tensor([0]).cuda())
    tags = torch.cat([torch.LongTensor([start_idx]).cuda(), tags])
    for i, unary_t in enumerate(unary):
        score = score + transitions[tags[i + 1], tags[i]] + unary_t[tags[i + 1]]
    score = score + transitions[end_idx, tags[-1]]
    return score


class RNNTaggerModel(nn.Module, Tagger):

    def save(self, outname):
        torch.save(self, outname)

    def get_criterion(self):
        return self.crit

    @staticmethod
    def load(outname, **kwargs):
        model = torch.load(outname)
        return model

    def _char_word_conv_embeddings(self, filtsz, char_dsz, wchsz, pdrop, unif):
        self.char_convs = []
        for fsz in filtsz:
            conv = nn.Sequential(
                pytorch_conv1d(char_dsz, wchsz, fsz, unif),
                pytorch_activation("relu")
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
            pytorch_activation("relu")
        ))

    def __init__(self):
        super(RNNTaggerModel, self).__init__()

    @staticmethod
    def create(labels, word_vec, char_vec, **kwargs):
        model = RNNTaggerModel()
        char_dsz = char_vec.dsz
        word_dsz = 0
        hsz = int(kwargs['hsz'])
        model.crf = bool(kwargs.get('crf', False))
        nlayers = int(kwargs.get('layers', 1))
        rnntype = kwargs.get('rnntype', 'lstm')
        print('RNN [%s]' % rnntype)
        unif = float(kwargs.get('unif', 0.25))
        wsz = kwargs.get('wsz', 30)
        filtsz = kwargs.get('cfiltsz')
        crf = bool(kwargs.get('crf', False))
        if crf:
            weights = torch.Tensor(len(labels), len(labels)).uniform_(-unif, unif)
            model.transitions = nn.Parameter(weights)
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
        model.rnn, out_hsz = pytorch_lstm(model.wchsz + word_dsz, hsz, rnntype, nlayers, pdrop, unif)
        model.decoder = nn.Sequential()
        append2seq(model.decoder, (
            nn.Dropout(pdrop),
            pytorch_linear(out_hsz, hsz, unif),
            pytorch_activation("tanh"),
            pytorch_linear(hsz, len(model.labels), unif)
        ))

        model.softmax = nn.LogSoftmax()
        model.crit = SequenceCriterion(len(labels))
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

    def _compute_unary_tb(self, x, xch):
        batchsz = xch.size(1)
        seqlen = xch.size(0)
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

    # Input better be xch, x
    def forward(self, input):
        START_IDX = self.labels.get("<GO>")
        END_IDX = self.labels.get("<EOS>")
        x = input[0].transpose(0, 1).contiguous()
        xch = input[1].transpose(0, 1).contiguous()
        lengths = input[2]
        batchsz = xch.size(1)
        seqlen = xch.size(0)

        probv = self._compute_unary_tb(x, xch)
        preds = []
        if self.crf is True:
            for pij, sl in zip(probv, lengths):
                unary = pij[:sl]
                viterbi, _ = viterbi_decode(unary, self.transitions, START_IDX, END_IDX)
                preds.append(viterbi)
        else:
            # Get batch (B, T)

            for pij, sl in zip(probv, lengths):
                _, unary = torch.max(pij[:sl], 1)
                preds.append(unary.data)

        return preds

    def compute_loss(self, input):
        START_IDX = self.labels.get("<GO>")
        END_IDX = self.labels.get("<EOS>")
        x = input[0].transpose(0, 1).contiguous()
        xch = input[1].transpose(0, 1).contiguous()
        lengths = input[2]
        tags = input[3]
        batchsz = xch.size(1)
        seqlen = xch.size(0)

        probv = self._compute_unary_tb(x, xch)
        batch_loss = 0.
        total_tags = 0.
        if self.crf is True:
            for pij, gold, sl in zip(probv, tags.data, lengths):

                gold_tags = gold[:sl]
                unary = pij[:sl]
                total_tags += len(gold_tags)
                forward_score = forward_algorithm_vec(unary, self.transitions, START_IDX, END_IDX)
                gold_score = score_sentence(unary, gold_tags, self.transitions, START_IDX, END_IDX)
                batch_loss += forward_score - gold_score
        else:
            # Get batch (B, T)
            for pij, gold, sl in zip(probv, tags, lengths):
                unary = pij[:sl]
                gold_tags = gold[:sl]
                total_tags += len(gold_tags)
                batch_loss += self.crit(unary, gold_tags)

        return batch_loss / total_tags

    def get_vocab(self, vocab_type='word'):
        return self.word_vocab if vocab_type == 'word' else self.char_vocab

    def get_labels(self):
        return self.labels

    def predict(self, x, xch, lengths):
        return predict_seq_bt(self, x, xch, lengths)


def create_model(labels, word_embedding, char_embedding, **kwargs):
    model = RNNTaggerModel.create(labels, word_embedding, char_embedding, **kwargs)
    return model
