import collections
from baseline.embeddings import register_embeddings, create_embeddings
from eight_mile.pytorch.embeddings import *
from eight_mile.pytorch.serialize import load_tlm_npz, load_tlm_output_npz, tlm_load_state_dict
from eight_mile.utils import read_config_stream, mime_type
from baseline.vectorizers import load_bert_vocab


class PyTorchEmbeddingsModel(PyTorchEmbeddings):
    """A subclass of embeddings layers to prep them for registration and creation via baseline.

    In tensorflow this layer handles the creation of placeholders and things like that so the
    embeddings layer can just be tensor in tensor out but in pytorch all it does is strip the
    unused `name` input and register them.
    """
    def __init__(self, _=None, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, model, name, **kwargs):
        kwargs.pop("dsz", None)
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)


@register_embeddings(name='default')
class LookupTableEmbeddingsModel(PyTorchEmbeddingsModel, LookupTableEmbeddings):
    pass


@register_embeddings(name='char-conv')
class CharConvEmbeddingsModel(PyTorchEmbeddingsModel, CharConvEmbeddings):
    pass


@register_embeddings(name='char-lstm')
class CharLSTMEmbeddingsModel(PyTorchEmbeddingsModel, CharLSTMEmbeddings):
    pass


@register_embeddings(name='char-transformer')
class CharTransformerEmbeddingsModel(PyTorchEmbeddingsModel, CharTransformerEmbeddings):
    pass


@register_embeddings(name='positional')
class PositionalLookupTableEmbeddingsModel(PyTorchEmbeddingsModel, PositionalLookupTableEmbeddings):
    pass


@register_embeddings(name='learned-positional')
class LearnedPositionalLookupTableEmbeddingsModel(PyTorchEmbeddingsModel, LearnedPositionalLookupTableEmbeddings):
    pass


@register_embeddings(name='learned-positional-w-bias')
class LearnedPositionalLookupTableEmbeddingsWithBiasModel(PyTorchEmbeddingsModel, LearnedPositionalLookupTableEmbeddingsWithBias):
    pass


@register_embeddings(name='bert-lookup-table-embeddings')
class BERTLookupTableEmbeddingsModel(PyTorchEmbeddingsModel, BERTLookupTableEmbeddings):
    pass


@register_embeddings(name='positional-char-conv')
class PositionalCharConvEmbeddingsModel(PyTorchEmbeddingsModel, PositionalCharConvEmbeddings):
    pass


@register_embeddings(name='learned-positional-char-conv')
class LearnedPositionalCharConvEmbeddingsModel(PyTorchEmbeddingsModel, LearnedPositionalCharConvEmbeddings):
    pass


@register_embeddings(name='positional-char-lstm')
class PositionalCharLSTMEmbeddingsModel(PyTorchEmbeddingsModel, PositionalCharLSTMEmbeddings):
    pass


@register_embeddings(name='learned-positional-char-lstm')
class LearnedPositionalCharLSTMEmbeddingsModel(PyTorchEmbeddingsModel, LearnedPositionalCharLSTMEmbeddings):
    pass


class TransformerLMEmbeddings(PyTorchEmbeddings):
    """Support embeddings trained with the TransformerLanguageModel class

    This method supports either subword or word embeddings, not characters

    """
    def __init__(self, **kwargs):
        super().__init__()
        # You dont actually have to pass this if you are using the `load_bert_vocab` call from your
        # tokenizer.  In this case, a singleton variable will contain the vocab and it will be returned
        # by `load_bert_vocab`
        # If you trained your model with MEAD/Baseline, you will have a `*.json` file which would want to
        # reference here
        vocab_file = kwargs.get('vocab_file')
        if vocab_file and os.path.exists(vocab_file):
            if vocab_file.endswith('.json'):
                self.vocab = read_config_stream(kwargs.get('vocab_file'))
            else:
                self.vocab = load_bert_vocab(kwargs.get('vocab_file'))
        else:
            self.vocab = kwargs.get('vocab', kwargs.get('known_vocab'))
            if self.vocab is None or isinstance(self.vocab, collections.Counter):
                self.vocab = load_bert_vocab(None)
        # When we reload, allows skipping restoration of these embeddings
        # If the embedding wasnt trained with token types, this allows us to add them later
        self.skippable = set(listify(kwargs.get('skip_restore_embeddings', [])))

        self.cls_index = self.vocab.get('[CLS]', self.vocab.get('<s>'))
        self.vsz = max(self.vocab.values()) + 1
        self.d_model = int(kwargs.get('dsz', kwargs.get('d_model', 768)))
        self.init_embed(**kwargs)
        self.proj_to_dsz = pytorch_linear(self.dsz, self.d_model) if self.dsz != self.d_model else _identity
        self.init_transformer(**kwargs)

    @property
    def dsz(self):
        return self.embeddings.output_dim

    def embed(self, input, token_type):
        return self.embeddings({'x': input, 'tt': token_type})

    def init_embed(self, **kwargs):
        # If you are using BERT, you probably want to use either
        # `learned-positional` with a token type feature
        # or `learned-positional-w-bias` if you dont care about the token type
        embed_type = kwargs.get('word_embed_type', 'learned-positional')
        x_embedding = create_embeddings(vsz=self.vsz, dsz=self.d_model, embed_type=embed_type, offset=kwargs.get("offset", 0),
                                        mxlen=kwargs.get('mxlen', 512))

        embeddings = {'x': x_embedding}
        # This is for BERT support when we are using 2 features
        token_type_vsz = kwargs.get('token_type_vsz')
        if token_type_vsz:
            tt_unif = float(kwargs.get('token_type_embed_unif', 0.001))
            tt_embedding = LookupTableEmbeddings(weights=torch.randn((token_type_vsz, self.d_model))*tt_unif,
                                                 padding_idx=None)
            embeddings['tt'] = tt_embedding
        # For bert, make sure this is `sum-layer-norm`
        reduction = kwargs.get('embeddings_reduction', kwargs.get('reduction', 'sum'))
        embeddings_dropout = kwargs.get('embeddings_dropout', 0.1)
        self.embeddings = EmbeddingsStack(embeddings, dropout_rate=embeddings_dropout, reduction=reduction)

    def init_transformer(self, **kwargs):
        num_layers = int(kwargs.get('layers', 12))
        num_heads = int(kwargs.get('num_heads', 12))
        pdrop = kwargs.get('dropout', 0.1)
        ff_pdrop = kwargs.get('ffn_dropout', 0.1)
        d_ff = int(kwargs.get('d_ff', 3072))
        d_k = kwargs.get('d_k')
        rpr_k = kwargs.get('rpr_k')
        layer_norms_after = kwargs.get('layer_norms_after', False)
        layer_norm_eps = float(kwargs.get('layer_norm_eps', 1e-12))
        activation = kwargs.get('activation', 'gelu')
        windowed_ra = kwargs.get('windowed_ra', False)
        rpr_value_on = kwargs.get('rpr_value_on', True)
        is_mlp = kwargs.get("mlp", False)
        if is_mlp:
            self.transformer = GatedMLPEncoderStack(self.d_model, pdrop=pdrop, layers=num_layers,
                                                    nctx=kwargs.get('nctx', 256),
                                                    activation=activation, ffn_pdrop=ff_pdrop,
                                                    layer_norm_eps=layer_norm_eps)
        else:

            self.transformer = TransformerEncoderStack(num_heads, d_model=self.d_model, pdrop=pdrop, scale=True,
                                                       layers=num_layers, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k,
                                                       activation=activation, ffn_pdrop=ff_pdrop,
                                                       layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps,
                                                       windowed_ra=windowed_ra, rpr_value_on=rpr_value_on)
        self.mlm = kwargs.get('mlm', True)
        self.finetune = kwargs.get('finetune', True)

    def forward(self, x, token_type=None):
        with torch.no_grad() if not self.finetune else contextlib.ExitStack():
            # the following line masks out the attention to padding tokens
            input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != Offsets.PAD, 1).unsqueeze(1).unsqueeze(1)
            # A causal LM should have a subsequent mask; and a masked LM should have no mask
            if not self.mlm:
                input_mask = input_mask & subsequent_mask(x.shape[1]).type_as(input_mask)
            embedding = self.embed(x, token_type)
            embedding = self.proj_to_dsz(embedding)
            transformer_out = self.transformer((embedding, input_mask))
            z = self.get_output(x, transformer_out)
            return z

    def get_output(self, inputs, z):
        return z

    def get_vocab(self):
        return self.vocab

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.d_model

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("tlm-words-embed", **kwargs)

        if embeddings.endswith('.bin'):
            # HuggingFace checkpoint, convert on the fly
            from eight_mile.pytorch.serialize import load_tlm_transformers_bin, BERT_HF_FT_LAYER_MAP
            unmatch = load_tlm_transformers_bin(c, embeddings, replace_layers=BERT_HF_FT_LAYER_MAP)
            if unmatch['missing'] or unmatch['unexpected']:
                raise Exception("Unable to load the HuggingFace checkpoint")
        if mime_type(embeddings) == 'application/zip' and not embeddings.endswith("pth"):
            keys_to_restore = set(list(c.embeddings.keys()))
            filtered_keys = keys_to_restore.difference(c.skippable)
            if not keys_to_restore:
                raise Exception("No keys to restore!")
            if len(filtered_keys) < len(keys_to_restore):
                logger.warning("Restoring only key [%s]", ' '.join(filtered_keys))
            load_tlm_npz(c, embeddings, filtered_keys)
        else:
            map_location = 'cpu' if kwargs.get('cpu_placement') else None
            tlm_load_state_dict(c, embeddings,
                                map_location=map_location)
        return c


@register_embeddings(name='tlm-words-embed')
class TransformerLMEmbeddingsModel(PyTorchEmbeddingsModel, TransformerLMEmbeddings):
    """Register embedding model for usage in mead"""
    pass


def _identity(x):
    return x


def _mean_pool(inputs, embeddings):
    mask = (inputs != Offsets.PAD)
    seq_lengths = mask.sum(1).float()
    embeddings = embeddings.masked_fill(mask.unsqueeze(-1) == False, 0.)
    return embeddings.sum(1)/seq_lengths.unsqueeze(-1)


def _max_pool(inputs, embeddings):
    mask = (inputs != Offsets.PAD)
    embeddings = embeddings.masked_fill(mask.unsqueeze(-1) == False, 0.)
    return torch.max(embeddings, 1, False)[0]


@register_embeddings(name='tlm-words-embed-pooled')
class TransformerLMPooledEmbeddingsModel(TransformerLMEmbeddingsModel):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.pooling = kwargs.get('pooling', 'cls').lower()
        reduction_pooling = kwargs.get('reduction_pooling', 'sqrt_length')
        reduction_d_k = kwargs.get('reduction_d_k', self.d_model)
        dropout = kwargs.get('dropout', 0.1)  # use the same dropout as the transformer encoder for reduction layer
        if self.pooling == 'max':
            self.pooling_op = _max_pool
        elif self.pooling == 'mean':
            self.pooling_op = _mean_pool
        elif self.pooling == 'sqrt_length':
            self.pooling_op = self._sqrt_length_pool

        elif self.pooling == "2ha":
            self.reduction_layer = TwoHeadConcat(self.d_model, dropout, scale=False, d_k=reduction_d_k)
            self.pooling_op = self._att_reduction

        elif self.pooling == "2ha_mean":
            self.reduction_layer = TwoHeadConcat(self.d_model, dropout, scale=False, d_k=reduction_d_k, pooling="mean")
            self.pooling_op = self._att_reduction

        elif self.pooling == "2ha_max":
            self.reduction_layer = TwoHeadConcat(self.d_model, dropout, scale=False, d_k=reduction_d_k, pooling="max")
            self.pooling_op = self._att_reduction

        elif self.pooling == "sha":
            self.reduction_layer = SingleHeadReduction(self.d_model, dropout, scale=False, d_k=reduction_d_k)
            self.pooling_op = self._att_reduction

        elif self.pooling == "sha_mean":
            self.reduction_layer = SingleHeadReduction(self.d_model, dropout, scale=False, d_k=reduction_d_k, pooling="mean")
            self.pooling_op = self._att_reduction

        elif self.pooling == "sha_max":
            self.reduction_layer = SingleHeadReduction(self.d_model, dropout, scale=False, d_k=reduction_d_k, pooling="max")
            self.pooling_op = self._att_reduction
        else:
            self.pooling_op = self._cls_pool

    def _att_reduction(self, inputs, embeddings):
        mask = (inputs != Offsets.PAD)
        att_mask = mask.unsqueeze(1).unsqueeze(1)
        reduced = self.reduction_layer((embeddings, embeddings, embeddings, att_mask))
        return reduced

    def get_dsz(self):
        if self.pooling.startswith('2ha'):
            return 2*self.d_model
        return self.d_model

    def _sqrt_length_pool(self, inputs, embeddings):
        mask = (inputs != Offsets.PAD)
        lengths = mask.sum(1)
        sqrt_length = lengths.float()
        embeddings = embeddings.masked_fill(mask.unsqueeze(-1) == False, 0.)
        embeddings = embeddings.sum(1) * sqrt_length.sqrt().unsqueeze(-1)
        return embeddings

    def _cls_pool(self, inputs, tensor):
        # Would prefer
        # tensor[inputs == self.cls_index]
        # but ONNX export fails
        B = tensor.shape[0]
        mask = (inputs == self.cls_index).unsqueeze(-1).expand_as(tensor)
        pooled = tensor.masked_select(mask).view(B, -1)
        return pooled

    def get_output(self, inputs, z):
        z = self.pooling_op(inputs, z)
        return z


@register_embeddings(name='tlm-words-embed-pooled-output')
class TransformerLMPooledEmbeddingsWithOutputModel(TransformerLMPooledEmbeddingsModel):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._output_dim = kwargs.get('output_dim', self.d_model)
        self.output_layer = pytorch_linear(self.d_model, self.output_dim)

    def get_dsz(self):
        return self._output_dim

    def get_output(self, inputs, z):
        z = self.pooling_op(inputs, z)
        return self.output_layer(z)

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("tlm-words-embed-pooled-output", **kwargs)

        if embeddings.endswith('.bin'):
            # HuggingFace checkpoint, convert on the fly
            from eight_mile.pytorch.serialize import load_tlm_transformers_bin, BERT_HF_FT_LAYER_MAP
            unmatch = load_tlm_transformers_bin(c, embeddings, replace_layers=BERT_HF_FT_LAYER_MAP)
            if unmatch['missing'] or unmatch['unexpected']:
                raise Exception("Unable to load the HuggingFace checkpoint")
        if mime_type(embeddings) == 'application/zip' and not embeddings.endswith("pth"):
            keys_to_restore = set(list(c.embeddings.keys()))
            filtered_keys = keys_to_restore.difference(c.skippable)
            if not keys_to_restore:
                raise Exception("No keys to restore!")
            if len(filtered_keys) < len(keys_to_restore):
                logger.warning("Restoring only key [%s]", ' '.join(filtered_keys))
            load_tlm_output_npz(c, embeddings, filtered_keys)
        else:
            map_location = 'cpu' if kwargs.get('cpu_placement') else None
            tlm_load_state_dict(c, embeddings,
                                str_map={'model.embeddings.embeddings.0.':'', 'model.output_layer': 'output_layer'},
                                map_location=map_location)
        return c

@register_embeddings(name='tlm-words-embed-pooled2d')
class TransformerLMPooled2DEmbeddingsModel(TransformerLMPooledEmbeddingsModel):

    def forward(self, xch, token_type=None):
        _0, _1, W = xch.shape
        pooled = super().forward(xch.view(-1, W))
        return pooled.view(_0, _1, self.get_dsz())
