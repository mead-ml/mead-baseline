from baseline.pytorch.torchy import vec_log_sum_exp
from baseline.pytorch.seq2seq import Seq2SeqModel
from eight_mile.utils import str2bool, write_yaml, read_yaml, Offsets
from eight_mile.pytorch.layers import *
import baseline.pytorch.embeddings
import baseline.embeddings
from baseline.progress import create_progress_bar
from torch.utils.data.dataset import IterableDataset, TensorDataset
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
    def __init__(self, model, warmup_steps=10000):
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
        self.score = nn.CosineSimilarity(dim=-1)
        self.model = model
        self.max_scale = math.sqrt(self.model.embedding_layers.get_dsz())
        self.steps = 0
        self.warmup_steps = warmup_steps

    def forward(self, inputs, targets):
        # This is the cosine distance annealing referred to in https://arxiv.org/pdf/1911.03688.pdf
        fract = min(self.steps / self.warmup_steps, 1)
        c = (self.max_scale-1) * fract + 1
        self.steps += 1
        # These will get broadcast to [B, B, H]
        query = self.model.encode_query(inputs).unsqueeze(1)  # [B, 1, H]
        response = self.model.encode_response(targets).unsqueeze(0)  # [1, B, H]
        # all_scores is now a batch x batch matrix where index (i, j) is the score between
        # the i^th x vector and the j^th y vector
        all_score = c * self.score(query, response)  # [B, B]
        # The diagonal has the scores of correct pair, (i, i)
        pos_score = torch.diag(all_score)
        # vec_log_sum_exp will calculate the batched log_sum_exp in a numerically stable way
        # the result is a [B, 1] vector which we squeeze to make it [B] to match the diag
        # Because we are minimizing the negative log we turned the division into a subtraction here
        loss = pos_score - vec_log_sum_exp(all_score, -1).squeeze()
        # Batch loss
        loss = torch.sum(loss)
        # minimize the negative loss
        return -loss


class PairedModel(nn.Module):

    def __init__(self, embeddings,
                 d_model,
                 d_ff,
                 dropout,
                 num_heads,
                 num_layers,
                 stacking_layers=None,
                 d_out=512,
                 d_k=64,
                 weight_std=0.02,
                 rpr_k=None):
        super().__init__()
        if stacking_layers is None:
            stacking_layers = [d_model] * 3

        self.weight_std = weight_std
        stacking_layers = listify(stacking_layers)
        transformer = TransformerEncoderStack(num_heads=num_heads, d_model=d_model,
                                              pdrop=dropout, layers=num_layers, activation='gelu', d_ff=d_ff,
                                              d_k=d_k, rpr_k=rpr_k)
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


class TransformerDiscriminator(nn.Module):

    def __init__(self, embeddings, d_model, d_ff, dropout, num_heads, num_layers, rpr_k, d_k, **kwargs):
        super().__init__()
        self.embeddings = EmbeddingsStack(embeddings, dropout)
        self.weight_std = kwargs.get('weight_std', 0.02)
        assert self.embeddings.dsz == d_model
        self.transformer = TransformerEncoderStack(num_heads, d_model=d_model, pdrop=dropout, scale=True,
                                                   layers=num_layers, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k)
        self.proj_to_output = pytorch_linear(d_model, 1)

        self.apply(self.init_layer_weights)
        self.lengths_feature = kwargs.get('lengths_feature', self.embeddings.keys()[0])

    def init_layer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.weight_std)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, features):
        embedded = self.embeddings(features)
        x = features[self.lengths_feature]
        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1).unsqueeze(1).unsqueeze(1)
        transformer_out = self.transformer((embedded, input_mask))
        binary = self.proj_to_output(transformer_out)
        return torch.sigmoid(binary)

    def create_loss(self):
        return nn.BCELoss()


class MultiFileLoader(IterableDataset):

    def __init__(self, directory, pattern, vocabs, vectorizer, nctx, last_turn_only=True):
        super().__init__()
        self.vectorizer = vectorizer
        self.pattern = pattern
        self.nctx = nctx
        self.directory = directory
        self.vocab = vocabs
        self.samples = 0
        self.rank = 0
        self.world_size = 1
        self.last_turn_only = last_turn_only
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

        if os.path.exists(f"{directory}/md.yml"):
            f = read_yaml(f"{directory}/md.yml")
            self.samples = f['num_samples']
        else:
            files = list(glob.glob(f"{directory}/{self.pattern}"))
            pg = create_progress_bar(len(files))
            for file in pg(files):
                with open(file) as rf:
                    for _ in rf:
                        self.samples += 1
            write_yaml({'num_samples': self.samples}, f"{directory}/md.yml")

    def __len__(self):
        return self.samples

    def __iter__(self):
        # Each node has the same worker_info, so the unique offsets for each is
        # rank * num_workers + worker_id
        # and the total available workers is world_size * num_workers
        worker_info = torch.utils.data.get_worker_info()
        files = sorted(list(glob.glob(f"{self.directory}/{self.pattern}")))

        if worker_info is None:
            num_workers_per_node = 1
            node_worker_id = 0
        else:
            num_workers_per_node = worker_info.num_workers
            node_worker_id = worker_info.id
        all_workers = (self.world_size * num_workers_per_node)
        files_per_worker = len(files) // all_workers
        offset = self.rank * num_workers_per_node + node_worker_id
        start_idx = offset * files_per_worker
        end_idx = start_idx + files_per_worker if offset < all_workers - 1 else len(files)
        print(f'worker {node_worker_id} [{start_idx}:{end_idx}]')

        self.vectorizer.mxlen = self.nctx

        for file in files[start_idx:end_idx]:
            with open(file) as rf:
                for line in rf:
                    response = self.process_line(line)
                    if response:
                        yield response

    def process_line(self, line):
        """Read in a line and turn it into an entry

        The entries will get collated by the data loader

        :param line:
        :return:
        """


class NextTurnPredictionFileLoader(MultiFileLoader):

    def process_line(self, line):
        pair = line.strip().split('\t')
        # Unfortunately, this occassionally happens, a bunch of blank turns etc.
        if len(pair) != 2:
            return None
        q, r = pair
        if q == '' or r == '':
            return None
        if self.last_turn_only:
            turns = q.split('<EOU>')
            q = turns[-1] if turns[-1].strip() != '' else turns[-2]
            if q.strip() == '':
                return None
            q_vec, q_valid_lengths = self.vectorizer.run(q.split(), self.vocab)
        else:
            q_vec, q_valid_lengths = self.vectorizer.run(reversed(q.split()), self.vocab)
            q_vec = np.roll(q_vec[::-1], -(self.vectorizer.mxlen - q_valid_lengths))

        r_vec, r_valid_lengths = self.vectorizer.run(r.split(), self.vocab)
        return q_vec, r_vec


class SequencePredictionFileLoader(MultiFileLoader):

    def process_line(self, line):
        line = line.strip()
        if not line:
            return None

        vec, valid_lengths = self.vectorizer.run(line.split(), self.vocab)
        if valid_lengths < 2:
            return None
        return vec, vec


class NextSequencePredictionFileLoader(MultiFileLoader):

    def process_line(self, line):
        line = line.strip()
        if not line:
            return None
        vec, valid_lengths = self.vectorizer.run(line.split(), self.vocab)
        if valid_lengths < 2:
            return None
        pair_entry_length = self.vectorizer.mxlen//2
        end_of_query = min(valid_lengths//2, pair_entry_length)
        # Front half is all tokens up until the half_way marker
        # Create a new query vector
        query = np.zeros(pair_entry_length, dtype=np.int)
        query[:end_of_query] = vec[:end_of_query]
        # Repurpose the existing vector as the response vector
        vec = vec[end_of_query:end_of_query+pair_entry_length]
        return query, vec


class MultiFileDatasetReader:
    """Provide a base-class to do operations that are independent of token representation
    """

    def __init__(self, nctx=64, model_file=None, vocab_file=None, pattern='*.txt', reader_type="ntp"):
        self.nctx = nctx
        self.pattern = pattern
        self.reader_type = reader_type
        self.vectorizer = BPEVectorizer1D(model_file=model_file, vocab_file=vocab_file, mxlen=nctx)

    def build_vocab(self, _=None):
        return {'x': self.vectorizer.vocab}

    def load(self, directory, vocabs):
        reader_type = self.reader_type.lower()
        if reader_type == "ntp":
            return NextTurnPredictionFileLoader(directory, self.pattern, vocabs, self.vectorizer, self.nctx)
        elif reader_type == "nsp":
            return NextSequencePredictionFileLoader(directory, self.pattern, vocabs, self.vectorizer, 2*self.nctx)
        else:
            print("Using files as an LM")
            return SequencePredictionFileLoader(directory, self.pattern, vocabs, self.vectorizer, self.nctx)


class TiedEmbeddingsSeq2SeqModel(Seq2SeqModel):

    def __init__(self, tied_embeddings, **kwargs):
        super().__init__({'x': tied_embeddings}, tied_embeddings, **kwargs)

    def input_tensor(self, key, batch_dict, perm_idx):
        tensor = batch_dict[key]
        tensor = self.drop_inputs(key, tensor)
        tensor = tensor[perm_idx]
        return tensor

    def make_input(self, batch_dict, perm=False):
        """Prepare the input.

        :param batch_dict: `dict`: The data.
        :param perm: `bool`: If True return the permutation index
            so that you can undo the sort if you want.
        """
        example = dict({})

        lengths = batch_dict[self.src_lengths_key]
        lengths, perm_idx = lengths.sort(0, descending=True)

        example['src_len'] = lengths
        for key in self.src_embeddings.keys():
            example[key] = self.input_tensor(key, batch_dict, perm_idx)

        if 'tgt' in batch_dict:
            tgt = batch_dict['tgt']
            example['dst'] = torch.cat([torch.full((tgt.shape[0], 1), Offsets.GO, device=tgt.device, dtype=tgt.dtype), tgt[:, :-1]], 1)
            example['tgt'] = tgt
            example['dst'] = example['dst'][perm_idx]
            example['tgt'] = example['tgt'][perm_idx]
        if perm:
            return example, perm_idx
        return example

    def create_loss(self, _=None):
        loss = super().create_loss()

        class LossFn(nn.Module):
            def __init__(self, model: nn.Module, l: nn.Module):
                super().__init__()
                self._loss = l
                self.model = model

            def forward(self, inputs, targets):
                lengths = torch.sum(inputs != 0, 1)
                in_ = self.model.make_input({"x": inputs, "x_lengths": lengths,  "tgt": targets})
                targets = in_['tgt']
                pred = self.model(in_)
                return self._loss(pred, targets)
        return LossFn(self, loss)
