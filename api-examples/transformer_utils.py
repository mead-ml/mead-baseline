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


def find_latest_checkpoint(checkpoint_dir: str, wildcard="checkpoint") -> str:
    step_num = 0
    for f in glob.glob(os.path.join(checkpoint_dir, f"{wildcard}*")):
        this_step_num = int(f.split("-")[-1])
        if this_step_num > step_num:
            checkpoint = f
            step_num = this_step_num
    return checkpoint

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


class DenseLN(nn.Module):
    """Dense (Linear) layer with optional activation given

    This module is the equivalent of the tf.keras.layer.Dense, module with optional activations applied
    """

    def __init__(
            self,
            insz: int,
            outsz: int,
            activation: Optional[str] = None,
            unif: float = 0,
            initializer: Optional[str] = None,
    ):
        """Constructor for "dense" or "linear" layer, with optional activation applied

        :param insz: The number of hidden units in the input
        :param outsz: The number of hidden units in the output
        :param activation: The activation function by name, defaults to `None`, meaning no activation is applied
        :param unif: An optional initialization value which can set the linear weights.  If given, biases will init to 0
        :param initializer: An initialization scheme by string name: `ortho`, `kaiming` or `he`, `xavier` or `glorot`
        """
        super().__init__()
        if insz == outsz:
            self.layer = SkipConnection(insz, activation)
        else:
            self.layer = Dense(insz, outsz, activation)

        self.output_dim = outsz
        self.ln = nn.LayerNorm(outsz, eps=1e-6)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run a linear projection over the input, followed by an optional activation given by constructor

        :param input: the input tensor
        :return: the transformed output
        """
        return self.ln(self.layer(input))


class DenseLNStack(nn.Module):
    """A stack of one or more hidden layers
    """

    def __init__(
            self,
            insz: int,
            hsz: Union[int, List[int]],
            activation: str = "gelu",
            pdrop_value: float = 0.5,
            init=None,
            skip_connect=True,
            **kwargs,
    ):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param insz: The number of input units
        :param hsz: The number of hidden units
        :param activation: The name of the activation function to use
        :param pdrop_value: The dropout probability
        :param init: The initializer
        :param skip_connect: whether use an overall skip connection
        """
        super().__init__()
        hszs = listify(hsz)
        self.output_dim = hsz[-1]
        current = insz
        layer_stack = []
        for hsz in hszs[:-1]:
            layer_stack.append(WithDropout(DenseLN(current, hsz, activation=activation), pdrop_value))
            current = hsz
        self.layer_stack = nn.Sequential(*layer_stack)
        self.final = Dense(current, self.output_dim)
        self.ln1 = nn.LayerNorm(insz, eps=1e-6)
        self.ln2 = nn.LayerNorm(self.output_dim, eps=1e-6)
        self.skip_connect = skip_connect
        if self.skip_connect:
            self.proj = Dense(insz, self.output_dim) if insz != self.output_dim else nn.Identity()
        self.requires_length = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param inputs: The fixed representation of the model

        :Keyword Arguments:
        * *hsz* -- (``int``) The number of hidden units (defaults to `100`)

        :return: The final layer
        """
        x = self.ln1(inputs)
        for layer in self.layer_stack:
            x = layer(x)
        x = self.final(x)
        if self.skip_connect:
            x = x + self.proj(inputs)
        return self.ln2(x)


class SingleHeadReduction(nn.Module):
    """
    Implementation of the "self_attention_head" layer from the conveRT paper
    """
    def __init__(
            self, d_model: int, dropout: float = 0.0, scale: bool = True, d_k: Optional[int] = None
    ):
        """
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()
        self.d_model = d_model
        if d_k is None:
            self.d_k = d_model
        else:
            self.d_k = d_k
        self.w_Q = Dense(d_model, self.d_k)
        self.w_K = Dense(d_model, self.d_k)
        if scale:
            self.attn_fn = SeqScaledDotProductAttention(dropout)
        else:
            self.attn_fn = SeqDotProductAttention(dropout)
        self.attn = None

    def forward(self, qkvm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """According to conveRT model's graph, they project token encodings to lower-dimensional query and key in single
        head, use them to calculate the attention score matrix that has dim [B, T, T], then sum over the query dim to
        get a tensor with [B, 1, T] (meaning the amount of attentions each token gets from all other tokens), scale it
        by sqrt of sequence lengths, then use it as the weight to weighted sum the token encoding to get the sentence
        encoding. we implement it in an equivalent way that can best make use of the eight_mile codes: do the matrix
        multiply with value first, then sum over the query dimension.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: sentence-level encoding with dim [B, d_model]
        """
        query, key, value, mask = qkvm
        batchsz = query.size(0)
        seq_mask = mask.squeeze()  # [B, T]
        seq_lengths = seq_mask.sum(dim=1)

        # (B, H, T, D), still have num_heads = 1 to use the attention function defined in eight_miles
        query = self.w_Q(query).view(batchsz, -1, 1, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, 1, self.d_k).transpose(1, 2)
        value = value.view(batchsz, -1, 1, self.d_model).transpose(1, 2)
        x = self.attn_fn((query, key, value, mask))  # [B, 1, T, D]
        self.attn = self.attn_fn.attn

        x = x.squeeze(1)  # [B, T, D]
        x = x * seq_mask.unsqueeze(-1)
        x = x.sum(dim=1)  # [B, D]
        x = x * seq_lengths.float().sqrt().unsqueeze(-1)
        return x


class TwoHeadConcat(nn.Module):
    """The 2-head attention layer of the conveRT model"""

    def __init__(self, d_model, dropout, scale=False, d_k=None):
        """Two parallel 1-head self-attention, then concatenate the output

        :param d_model: dim of the self-attention
        :param dropout: dropout of the self-attention
        :param scale: scale fo the self-attention
        :param d_k: d_k of the self-attention

        :return: concatenation of the two 1-head attention
        """
        super().__init__()
        self.reduction1 = SingleHeadReduction(d_model, dropout, scale=scale, d_k=d_k)
        self.reduction2 = SingleHeadReduction(d_model, dropout, scale=scale, d_k=d_k)

    def forward(self, inputs: torch.Tensor):
        x = inputs
        encoding1 = self.reduction1(x)
        encoding2 = self.reduction2(x)
        x = torch.cat([encoding1, encoding2], dim=-1)
        return x


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
                 rpr_k=None,
                 ff_pdrop=0.2):
        super().__init__()
        if stacking_layers is None:
            stacking_layers = [d_model] * 3

        self.weight_std = weight_std
        stacking_layers = listify(stacking_layers)
        transformer = TransformerEncoderStack(num_heads=num_heads, d_model=d_model,
                                              pdrop=dropout, layers=num_layers, activation='gelu', d_ff=d_ff,
                                              d_k=d_k, rpr_k=rpr_k)
        self.attention_layer = TwoHeadConcat(d_model, dropout, scale=True, d_k=d_k)
        self.transformer_layers = transformer
        self.embedding_layers = embeddings
        self.ff1 = DenseLNStack(2*d_model, stacking_layers + [d_out], activation='gelu', pdrop_value=ff_pdrop)
        self.ff2 = DenseLNStack(2*d_model, stacking_layers + [d_out], activation='gelu', pdrop_value=ff_pdrop)
        self.apply(self.init_layer_weights)

    def init_layer_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.weight_std)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def encode_query(self, query):
        query_mask = (query != Offsets.PAD)
        att_mask = query_mask.unsqueeze(1).unsqueeze(1)
        embedded = self.embedding_layers(query)
        encoded_query = self.transformer_layers((embedded, att_mask))
        encoded_query = self.attention_layer((encoded_query, encoded_query, encoded_query, att_mask))
        encoded_query = self.ff1(encoded_query)
        return encoded_query

    def encode_response(self, response):
        response_mask = (response != Offsets.PAD)
        att_mask = response_mask.unsqueeze(1).unsqueeze(1)
        embedded = self.embedding_layers(response)
        encoded_response = self.transformer_layers((embedded, att_mask))
        encoded_response = self.attention_layer((encoded_response, encoded_response, encoded_response, att_mask))
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
        class Loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss = nn.BCELoss()

            def forward(self, input, target):
                fake_loss = self.loss(input[target == 0], target[target == 0])
                real_loss = self.loss(input[target != 0], target[target != 0])
                return real_loss + fake_loss
        return Loss()


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
