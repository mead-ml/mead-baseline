from baseline.pytorch.torchy import vec_log_sum_exp
from eight_mile.utils import str2bool, write_json, Offsets
from eight_mile.pytorch.layers import *
import baseline.pytorch.embeddings
import baseline.embeddings
from torch.utils.data import TensorDataset, Dataset
from baseline.vectorizers import Token1DVectorizer, BPEVectorizer1D
import codecs
from collections import Counter
import glob

class TripletLoss(nn.Module):
    """Provide a Triplet Loss using the reversed batch for negatives"""
    def __init__(self, model):
        super().__init__()
        self.score = nn.CosineSimilarity(dim=1)
        self.model = model

    def forward(self, inputs, targets):
        # reverse the batch and use as a negative example
        neg = targets.flip(0)
        query = self.model.encode_query(inputs)
        response = self.model.encode_response(targets)
        neg_response = self.model.encode_response(neg)
        pos_score = self.score(query, response)
        neg_score = self.score(query, neg_response)
        score = neg_score - pos_score
        score = score.masked_fill(score < 0.0, 0.0).sum(0)
        return score


class AllLoss(nn.Module):
    def __init__(self, model):
        r"""Loss from here https://arxiv.org/pdf/1705.00652.pdf see section 4

        We want to minimize the negative log prob of y given x

        -log P(y|x)

        P(y|x) P(x) = P(x, y)                             Chain Rule of Probability
        P(y|x) = P(x, y) / P(x)                           Algebra
        P(y|x) = P(x, y) / \sum_\hat(y) P(x, y = \hat(y)) Marginalize over all possible ys to get the probability of x
        P_approx(y|x) = P(x, y) / \sum_i^k P(x, y_k)      Approximate the Marginalization by just using the ys in the batch

        S(x, y) is the score (cosine similarity between x and y in this case) from our neural network
        P(x, y) = e^S(x, y)

        P(y|x) = e^S(x, y) / \sum_i^k e^S(x, y_k)
        log P(y|x) = log( e^S(x, y) / \sum_i^k e^S(x, y_k))
        log P(y|x) = S(x, y) - log \sum_i^k e^S(x, y_k)
        -log P(y|x) = -(S(x, y) - log \sum_i^k e^S(x, y_k))
        """
        super().__init__()
        self.c = 1.0
        self.score = nn.CosineSimilarity(dim=-1)
        self.model = model
        self.max_scale = math.sqrt(self.model.embedding_layers.get_dsz())

    def forward(self, inputs, targets):
        # These will get broadcast to [B, B, H]
        query = self.model.encode_query(inputs).unsqueeze(1)  # [B, 1, H]
        response = self.model.encode_response(targets).unsqueeze(0)  # [1, B, H]
        # all_scores is now a batch x batch matrix where index (i, j) is the score between
        # the i^th x vector and the j^th y vector
        all_score = self.max_scale * self.score(query, response) # [B, B]
        # The diagonal has the scores of correct pair, (i, i)
        pos_score = torch.diag(all_score)
        # vec_log_sum_exp will calculate the batched log_sum_exp in a numerically stable way
        # the result is a [B, 1] vector which we squeeze to make it [B] to match the diag
        # Because we are minimizing the negative log we turned the division into a subtraction here
        loss = pos_score - vec_log_sum_exp(all_score, -1).squeeze()
        # Batch loss
        loss = torch.mean(loss)
        # minimize the negative loss
        return -loss


class PairedModel(nn.Module):

    def __init__(self, embeddings, d_model, d_ff, dropout, num_heads, num_layers, stacking_layers=None, d_out=512, d_k=64, weight_std=0.02):
        super().__init__()
        if stacking_layers is None:
            stacking_layers = [d_model] * 3

        self.weight_std = weight_std
        stacking_layers = listify(stacking_layers)
        transformer = TransformerEncoderStack(num_heads=num_heads, d_model=d_model,
                                              pdrop=dropout, layers=num_layers, activation='gelu', d_ff=d_ff, d_k=d_k)
        self.attention_layer = MultiHeadedAttention(2, d_model, dropout, scale=False, d_k=d_k)
        self.transformer_layers = transformer
        self.embedding_layers = embeddings
        self.ff1 = DenseStack(d_model, stacking_layers + [d_out], activation='gelu')
        self.ff2 = DenseStack(d_model, stacking_layers + [d_out], activation='gelu')
        self.apply(self.init_layer_weights)

    def init_layer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.weight_std)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def encode_query(self, query):
        query_mask = (query != Offsets.PAD)
        query_length = query_mask.sum(-1)
        query_mask = query_mask.unsqueeze(1).unsqueeze(1)
        embedded = self.embedding_layers(query)
        encoded_query = self.transformer_layers((embedded, query_mask))
        encoded_query = self.attention_layer((encoded_query, encoded_query, encoded_query, query_mask))
        encoded_query = encoded_query.sum(1) / query_length.float().sqrt().unsqueeze(1)
        encoded_query = self.ff1(encoded_query)
        return encoded_query

    def encode_response(self, response):
        response_mask = (response != Offsets.PAD)
        response_length = response_mask.sum(-1)
        response_mask = response_mask.unsqueeze(1).unsqueeze(1)
        embedded = self.embedding_layers(response)
        encoded_response = self.transformer_layers((embedded, response_mask))
        encoded_response = self.attention_layer((encoded_response, encoded_response, encoded_response, response_mask))
        encoded_response = encoded_response.sum(1) / response_length.float().sqrt().unsqueeze(1)
        encoded_response = self.ff2(encoded_response)

        return encoded_response

    def forward(self, query, response):
        encoded_query = self.encode_query(query)
        encoded_response = self.encode_response(response)
        return encoded_query, encoded_response

    def create_loss(self, loss_type='all'):
        if loss_type == 'all':
            return AllLoss(self)
        return TripletLoss(self)


class SingleSourceTensorDatasetReaderBase(object):
    """Provide a base-class to do operations that are independent of token representation
    """
    def __init__(self, nctx, vectorizers):
        self.vectorizers = vectorizers
        self.nctx = nctx
        self.num_words = {}

    def build_vocab(self, files):
        vocabs = {k: Counter() for k in self.vectorizers.keys()}

        for file in files:
            if file is None:
                continue
            self.num_words[file] = 0
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                sentences = []
                for line in f:
                    split_sentence = line.split() + ['<EOS>']
                    self.num_words[file] += len(split_sentence)
                    sentences += split_sentence
                for k, vectorizer in self.vectorizers.items():
                    vocabs[k].update(vectorizer.count(sentences))
        return vocabs

    def load_features(self, filename, vocabs):

        features = dict()
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            sentences = []
            for line in f:
                sentences += line.strip().split() + ['<EOS>']
            for k, vectorizer in self.vectorizers.items():
                vec, valid_lengths = vectorizer.run(sentences, vocabs[k])
                features[k] = vec[:valid_lengths]
                shp = list(vectorizer.get_dims())
                shp[0] = valid_lengths
                features['{}_dims'.format(k)] = tuple(shp)
        return features


class SingleSourceTensorWordDatasetReader(SingleSourceTensorDatasetReaderBase):
    """Read each word, and produce a tensor of x and y that are identical
    """
    def __init__(self, nctx: int, use_subword: bool = False, model_file: str = None, vocab_file: str = None):
        """Create a reader with a context window that reads words

        :param nctx: The context window length
        :param use_subword: If true, use BPE, else words
        """
        self.use_subword = use_subword
        if self.use_subword:
            vectorizer = BPEVectorizer1D(model_file=model_file, vocab_file=vocab_file)
        else:
            vectorizer = Token1DVectorizer(transform_fn=baseline.lowercase)
        super().__init__(nctx, {'x': vectorizer})

    def build_vocab(self, files):
        """Read the vocab file to get the tokens

        :param files:
        :return:
        """
        if self.use_subword:
            super().build_vocab(files)
            return {'x': self.vectorizers['x'].vocab}
        return super().build_vocab(files)

    def load(self, filename, vocabs):
        features = self.load_features(filename, vocabs)
        x_tensor = torch.tensor(features['x'], dtype=torch.long)
        batch_width = self.nctx * 2
        num_sequences_word = (x_tensor.size(0) // batch_width) * batch_width
        x_tensor = x_tensor.narrow(0, 0, num_sequences_word).view(-1, batch_width)
        # Take the first half for x_tensor, and the second half for y_tensor
        return TensorDataset(x_tensor[:, :self.nctx], x_tensor[:, self.nctx:])

