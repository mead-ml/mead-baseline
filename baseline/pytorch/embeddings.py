from baseline.embeddings import register_embeddings, create_embeddings
from eight_mile.pytorch.embeddings import *
from eight_mile.pytorch.serialize import load_tlm_npz
from eight_mile.utils import read_json, mime_type
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
        if vocab_file and vocab_file.endswith('.json'):
            self.vocab = read_json(kwargs.get('vocab_file'))
        else:
            self.vocab = load_bert_vocab(kwargs.get('vocab_file'))
        self.cls_index = self.vocab['[CLS]']
        self.vsz = max(self.vocab.values()) + 1
        self.d_model = int(kwargs.get('dsz', kwargs.get('d_model', 768)))
        self.init_embed(**kwargs)
        self.proj_to_dsz = pytorch_linear(self.dsz, self.d_model) if self.dsz != self.d_model else _identity
        self.init_transformer(**kwargs)

    @property
    def dsz(self):
        return self.embeddings.output_dim

    def embed(self, input):
        return self.embeddings({'x': input})

    def init_embed(self, **kwargs):
        # If you are using BERT, you probably want to use either
        # `learned-positional` with a token type feature
        # or `learned-positional-w-bias` if you dont care about the token type
        embed_type = kwargs.get('word_embed_type', 'learned-positional')
        x_embedding = create_embeddings(vsz=self.vsz, dsz=self.d_model, embed_type=embed_type)

        embeddings = {'x': x_embedding}
        # This is for BERT support when we are using 2 features
        #token_type_vsz = kwargs.get('token_type_vsz')
        #if token_type_vsz:
        #    tt_embedding = LookupTableEmbeddings(vsz=token_type_vsz, dsz=self.dsz)
        #    embeddings['tt'] = tt_embedding
        # For bert, make sure this is `sum-layer-norm`
        reduction = kwargs.get('embeddings_reduction', kwargs.get('reduction'))
        embeddings_dropout = kwargs.get('embeddings_dropout', 0.1)
        self.embeddings = EmbeddingsStack(embeddings, dropout_rate=embeddings_dropout, reduction=reduction)

    def init_transformer(self, **kwargs):
        num_layers = int(kwargs.get('layers', 12))
        num_heads = int(kwargs.get('num_heads', 12))
        pdrop = kwargs.get('dropout', 0.1)
        d_ff = int(kwargs.get('d_ff', 3072))
        d_k = kwargs.get('d_k')
        rpr_k = kwargs.get('rpr_k')
        layer_norms_after = kwargs.get('layer_norms_after', False)
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        self.transformer = TransformerEncoderStack(num_heads, d_model=self.d_model, pdrop=pdrop, scale=True,
                                                   layers=num_layers, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k,
                                                   layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps)
        self.mlm = kwargs.get('mlm', False)
        self.finetune = kwargs.get('finetune', True)

    def _model_mask(self, nctx):
        """This function creates the mask that controls which token to be attended to depending on the model. A causal
        LM should have a subsequent mask; and a masked LM should have no mask."""
        if self.mlm:
            return torch.ones((1, 1, nctx, nctx), dtype=torch.long)
        else:
            return subsequent_mask(nctx)

    def forward(self, x):
        # the following line masks out the attention to padding tokens
        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1).unsqueeze(1).unsqueeze(1)
        # the following line builds mask depending on whether it is a causal lm or masked lm
        input_mask = input_mask & self._model_mask(x.shape[1]).type_as(input_mask)
        embedding = self.embed(x)
        embedding = self.proj_to_dsz(embedding)
        transformer_out = self.transformer((embedding, input_mask))
        z = self.get_output(x, transformer_out)
        return z

    def get_output(self, inputs, z):
        return z if self.finetune else z.detach()

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
        if mime_type(embeddings) == 'application/zip':
            load_tlm_npz(c, embeddings)
        else:
            unmatch = c.load_state_dict(torch.load(embeddings), strict=False)
            if unmatch.missing_keys or len(unmatch.unexpected_keys) > 2:
                print("Warning: Embedding doesn't match with the checkpoint being loaded.")
                print(f"missing keys: {unmatch.missing_keys}\n unexpected keys: {unmatch.unexpected_keys}")
        return c


@register_embeddings(name='tlm-words-embed')
class TransformerLMEmbeddingsModel(PyTorchEmbeddingsModel, TransformerLMEmbeddings):
    """Register embedding model for usage in mead"""
    pass


def _identity(x):
    return x


def _mean_pool(inputs, embeddings):
    mask = (inputs != 0)
    embeddings = embeddings * mask.unsqueeze(-1)
    return torch.mean(embeddings, 1, False)


def _max_pool(inputs, embeddings):
    mask = (inputs != 0)
    embeddings = embeddings * mask.unsqueeze(-1)
    return torch.max(embeddings, 1, False)[0]


@register_embeddings(name='tlm-words-embed-pooled')
class TransformerLMPooledEmbeddingsModel(TransformerLMEmbeddingsModel):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        pooling = kwargs.get('pooling', 'cls')
        if pooling == 'max':
            self.pooling_op = _max_pool
        elif pooling == 'mean':
            self.pooling_op = _mean_pool
        elif pooling == 'sqrt_length':
            self.pooling_op = self._sqrt_length_pool
        else:
            self.pooling_op = self._cls_pool

    def _sqrt_length_pool(self, inputs, embeddings):
        mask = (inputs != 0)
        lengths = mask.sum(1)
        sqrt_length = lengths.float().sqrt().unsqueeze(1)
        embeddings = embeddings * mask.unsqueeze(-1)
        embeddings = embeddings.sum(1) / sqrt_length
        return embeddings

    def _cls_pool(self, inputs, tensor):
        pooled = tensor[inputs == self.cls_index]
        return pooled

    def get_output(self, inputs, z):
        z = self.pooling_op(inputs, z)
        return z if self.finetune else z.detach()
