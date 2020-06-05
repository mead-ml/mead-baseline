import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from eight_mile.pytorch.layers import Dense, TransformerEncoderStack, TransformerEncoder, EmbeddingsStack
from eight_mile.pytorch.embeddings import LookupTableEmbeddings, LearnedPositionalLookupTableEmbeddingsWithBias

# BERT HuggingFace Tokenizers checkpoints can be converted into MEAD Baseline Transformer checkpoints
# With a simple name change
BERT_HF_LAYER_MAP = {
    ## FFN weights
    'bert.encoder.layer.{}.intermediate.dense.weight': 'generator.encoders.{}.ffn.0.layer.weight',
    'bert.encoder.layer.{}.intermediate.dense.bias': 'generator.encoders.{}.ffn.0.layer.bias',
    'bert.encoder.layer.{}.output.dense.weight': 'generator.encoders.{}.ffn.3.layer.weight',
    'bert.encoder.layer.{}.output.dense.bias': 'generator.encoders.{}.ffn.3.layer.bias',

    ## MHA weights
    'bert.encoder.layer.{}.attention.self.key.weight': 'generator.encoders.{}.self_attn.w_K.layer.weight',
    'bert.encoder.layer.{}.attention.self.key.bias': 'generator.encoders.{}.self_attn.w_K.layer.bias',
    'bert.encoder.layer.{}.attention.self.query.weight': 'generator.encoders.{}.self_attn.w_Q.layer.weight',
    'bert.encoder.layer.{}.attention.self.query.bias': 'generator.encoders.{}.self_attn.w_Q.layer.bias',
    'bert.encoder.layer.{}.attention.self.value.weight': 'generator.encoders.{}.self_attn.w_V.layer.weight',
    'bert.encoder.layer.{}.attention.self.value.bias': 'generator.encoders.{}.self_attn.w_V.layer.bias',
    'bert.encoder.layer.{}.attention.output.dense.weight': 'generator.encoders.{}.self_attn.w_O.layer.weight',
    'bert.encoder.layer.{}.attention.output.dense.bias': 'generator.encoders.{}.self_attn.w_O.layer.bias',

    ## LN weights
    # The names in of layer norm our transformers are a bit unspecific
    # think of ln1 as ln_x and ln2 as ln_attn_output
    'bert.encoder.layer.{}.output.LayerNorm.beta': 'generator.encoders.{}.ln1.bias',
    'bert.encoder.layer.{}.output.LayerNorm.gamma': 'generator.encoders.{}.ln1.weight',
    'bert.encoder.layer.{}.attention.output.LayerNorm.beta': 'generator.encoders.{}.ln2.bias',
    'bert.encoder.layer.{}.attention.output.LayerNorm.gamma': 'generator.encoders.{}.ln2.weight'
}

BERT_HF_FT_LAYER_MAP = {
    ## FFN weights
    'bert.encoder.layer.{}.intermediate.dense.weight': 'transformer.encoders.{}.ffn.0.layer.weight',
    'bert.encoder.layer.{}.intermediate.dense.bias': 'transformer.encoders.{}.ffn.0.layer.bias',
    'bert.encoder.layer.{}.output.dense.weight': 'transformer.encoders.{}.ffn.3.layer.weight',
    'bert.encoder.layer.{}.output.dense.bias': 'transformer.encoders.{}.ffn.3.layer.bias',

    ## MHA weights
    'bert.encoder.layer.{}.attention.self.key.weight': 'transformer.encoders.{}.self_attn.w_K.layer.weight',
    'bert.encoder.layer.{}.attention.self.key.bias': 'transformer.encoders.{}.self_attn.w_K.layer.bias',
    'bert.encoder.layer.{}.attention.self.query.weight': 'transformer.encoders.{}.self_attn.w_Q.layer.weight',
    'bert.encoder.layer.{}.attention.self.query.bias': 'transformer.encoders.{}.self_attn.w_Q.layer.bias',
    'bert.encoder.layer.{}.attention.self.value.weight': 'transformer.encoders.{}.self_attn.w_V.layer.weight',
    'bert.encoder.layer.{}.attention.self.value.bias': 'transformer.encoders.{}.self_attn.w_V.layer.bias',
    'bert.encoder.layer.{}.attention.output.dense.weight': 'transformer.encoders.{}.self_attn.w_O.layer.weight',
    'bert.encoder.layer.{}.attention.output.dense.bias': 'transformer.encoders.{}.self_attn.w_O.layer.bias',

    ## LN weights
    # The names in of layer norm our transformers are a bit unspecific
    # think of ln1 as ln_x and ln2 as ln_attn_output
    'bert.encoder.layer.{}.output.LayerNorm.beta': 'transformer.encoders.{}.ln1.bias',
    'bert.encoder.layer.{}.output.LayerNorm.gamma': 'transformer.encoders.{}.ln1.weight',
    'bert.encoder.layer.{}.attention.output.LayerNorm.beta': 'transformer.encoders.{}.ln2.bias',
    'bert.encoder.layer.{}.attention.output.LayerNorm.gamma': 'transformer.encoders.{}.ln2.weight'
}

BERT_HF_EMBED_MAP = {
    ## Embedding weights
    'bert.embeddings.word_embeddings.weight': 'embeddings.embeddings.0.embeddings.weight',
    'bert.embeddings.position_embeddings.weight': 'embeddings.embeddings.0.pos_embeddings.weight',
    'bert.embeddings.token_type_embeddings.weight': 'embeddings.embeddings.1.embeddings.weight',
    'bert.embeddings.LayerNorm.beta': 'embeddings.reduction.ln.bias',
    'bert.embeddings.LayerNorm.gamma': 'embeddings.reduction.ln.weight',
}


def convert_transformers_keys(num_layers: int, d: Dict, replace_layer_map: Dict = BERT_HF_LAYER_MAP, replace_embed_map: Dict = BERT_HF_EMBED_MAP) -> Dict:
    m = {}
    for i in range(num_layers):
        for k, v in replace_layer_map.items():
            m[v.format(i)] = d[k.format(i)]

    for k, v in replace_embed_map.items():
        m[v] = d[k]


    return m


def tlm_load_state_dict(module: nn.Module, checkpoint_file: str):
    """

    :param tlm: Safely loads the state dict for a transformer encoder
    :param checkpoint_file: The file name
    :return: None
    """
    from_str = 'transformer'
    to_str = 'generator'
    if hasattr(module, 'transformer'):
        from_str = 'generator'
        to_str = 'transformer'
    ckpt_dict = torch.load(checkpoint_file)
    renamed = {}
    for k, v in ckpt_dict.items():
        renamed[k.replace(from_str, to_str)] = v
    unmatch = module.load_state_dict(renamed, strict=False)
    if unmatch.missing_keys or len(unmatch.unexpected_keys) > 2:
        print("Warning: Embedding doesn't match with the checkpoint being loaded.")
        print(f"missing keys: {unmatch.missing_keys}\n unexpected keys: {unmatch.unexpected_keys}")


def to_weight_array(pytorch_layer: nn.Module, name: str) -> Dict:
    """Convert a {`LayerNorm`, `Linear`, `layers.Dense`} to `weights` and `bias` arrays

    :param pytorch_layer: A layer to get weights for
    :param name: The name of this layer to serialize
    :return: A Dictionary containing `weights` and `bias` keys
    """
    if isinstance(pytorch_layer, Dense):
        pytorch_layer = pytorch_layer.layer
    weights = pytorch_layer.weight.cpu().detach().numpy()
    bias = pytorch_layer.bias.cpu().detach().numpy()
    return {f"{name}/weights": weights, f"{name}/bias": bias}


def from_weight_array(pytorch_layer: nn.Module, d: Dict, name: str):
    """Read in {`LayerNorm`, `Linear`, `layers.Dense`} from `weights` and `bias` fields

    :param pytorch_layer: A layer to get weights for
    :param d: A Dict containing the arrays by key
    :param name: The name of this layer
    :return: None
    """
    if isinstance(pytorch_layer, Dense):
        pytorch_layer = pytorch_layer.layer
    device = pytorch_layer.weight.device
    pytorch_layer.weight = nn.Parameter(torch.from_numpy(d[f"{name}/weights"]).to(device=device), requires_grad=True)
    pytorch_layer.bias = nn.Parameter(torch.from_numpy(d[f"{name}/bias"]).to(device=device), requires_grad=True)


def to_ffn_array(pytorch_ffn: nn.Sequential, name: str) -> Dict:
    """Convert a `FFN` layer to a set of arrays

    :param pytorch_ffn: An `FFN` layer for Transformers
    :param name: The name of the layer
    :return: A Dict containing the arrays by key
    """
    d = {}
    d.update(to_weight_array(pytorch_ffn[0], f"{name}/expansion"))
    d.update(to_weight_array(pytorch_ffn[3], f"{name}/squeeze"))
    return d


def from_ffn_array(pytorch_ffn: nn.Sequential, d: Dict, name: str):
    """Restore an `FFN` layer's weights from a set of arrays

    :param pytorch_ffn: An `FFN` layer for Transformers
    :param d: A Dict containing the arrays by key
    :param name: The name of the layer
    :return: None
    """
    from_weight_array(pytorch_ffn[0], d, f"{name}/expansion")
    from_weight_array(pytorch_ffn[3], d, f"{name}/squeeze")


def to_attn_array(pytorch_attn: nn.Module, name: str) -> Dict:
    """Convert a self-attention module to a set of arrays

    :param pytorch_attn: The self-attention layer of the transformer encoder, could be MultiHeadedAttention or
    MultiHeadedRelativeAttention
    :param name: The name of the layer
    :return: A Dict containing the arrays by key
    """
    d = {}

    d.update(to_weight_array(pytorch_attn.w_Q, f"{name}/w_Q"))
    d.update(to_weight_array(pytorch_attn.w_K, f"{name}/w_K"))
    d.update(to_weight_array(pytorch_attn.w_V, f"{name}/w_V"))
    d.update(to_weight_array(pytorch_attn.w_O, f"{name}/w_O"))

    if hasattr(pytorch_attn, 'rpr_key'):
        rpr_key_weights = pytorch_attn.rpr_key.weight.cpu().detach().numpy()
        rpr_value_weights = pytorch_attn.rpr_value.weight.cpu().detach().numpy()
        d.update({f"{name}/rpr_key": rpr_key_weights})
        d.update({f"{name}/rpr_value": rpr_value_weights})

    return d


def from_attn_array(pytorch_attn: nn.Module, d: Dict, name: str):
    """Restore the self-attention module from a set of arrays

    :param pytorch_attn: A self-attention module, could be MultiHeadedAttention or MultiHeadedRelativeAttention
    :param d: A Dict of arrays by key
    :param name: The name of the layer
    """
    from_weight_array(pytorch_attn.w_Q, d, f"{name}/w_Q")
    from_weight_array(pytorch_attn.w_K, d, f"{name}/w_K")
    from_weight_array(pytorch_attn.w_V, d, f"{name}/w_V")
    from_weight_array(pytorch_attn.w_O, d, f"{name}/w_O")

    if hasattr(pytorch_attn, 'rpr_key'):
        device = pytorch_attn.rpr_key.weight.device
        pytorch_attn.rpr_key.weight = torch.nn.Parameter(torch.from_numpy(d[f"{name}/rpr_key"]).to(device=device))
        pytorch_attn.rpr_value.weight = torch.nn.Parameter(torch.from_numpy(d[f"{name}/rpr_value"]).to(device=device))


def to_encoder_array(pytorch_encoder: TransformerEncoder, name: str) -> Dict:
    """Convert a `TransformerEncoder` layer to an set of numpy arrays

    :param pytorch_encoder: A `TransformerEncoder` layer
    :param name: The layer name
    :return: A Dict of arrays by key
    """
    d = {}
    d.update(to_weight_array(pytorch_encoder.ln1, f"{name}/ln1"))
    d.update(to_weight_array(pytorch_encoder.ln2, f"{name}/ln2"))
    d.update(to_attn_array(pytorch_encoder.self_attn, f"{name}/attn"))
    d.update(to_ffn_array(pytorch_encoder.ffn, f"{name}/ffn"))
    return d


def from_encoder_array(pytorch_encoder: TransformerEncoder, d: Dict, name: str):
    """Restore a `TransformerEncoder` layer from a set of numpy arrays

    :param pytorch_encoder: A `TransformerEncoder` layer
    :param d: A Dict of arrays by key
    :param name: The layer name
    :return: None
    """
    from_weight_array(pytorch_encoder.ln1, d, f"{name}/ln1")
    from_weight_array(pytorch_encoder.ln2, d, f"{name}/ln2")
    from_attn_array(pytorch_encoder.self_attn, d, f"{name}/attn")
    from_ffn_array(pytorch_encoder.ffn, d, f"{name}/ffn")


def to_embed_array(pytorch_embed: nn.Module, name: str) -> Dict:
    """Convert positional embedding to a `weights` array, if it's learned positional embedding,
    save the pos_weight as well

    :param pytorch_embed: An embedding module
    :param name: A layer name
    :return: A Dict containing the embedding `weights`
    """
    d = {}
    weights = pytorch_embed.embeddings.weight.cpu().detach().numpy()
    d.update({f"{name}/weights": weights})
    if hasattr(pytorch_embed, 'pos_embeddings'):
        pos_weights = pytorch_embed.pos_embeddings.weight.cpu().detach().numpy()
        d.update({f"{name}/pos_weights": pos_weights})
    return d


def from_embed_array(pytorch_embed: nn.Module, d: Dict, name: str):
    """Restore a positional embedding from a `weights` array, if it's a learned positional embedding, the pos_weights
    is also restored

    :param pytorch_embed: An embedding module
    :param d: A Dict containing a `weights` array to restore
    :param name: name of the layer
    :return: None
    """
    device = pytorch_embed.embeddings.weight.device
    pytorch_embed.embeddings.weight = torch.nn.Parameter(torch.from_numpy(d[f"{name}/weights"]).to(device=device), requires_grad=True)
    if hasattr(pytorch_embed, 'pos_embeddings'):
        pytorch_embed.pos_embeddings.weight = torch.nn.Parameter(torch.from_numpy(d[f"{name}/pos_weights"]).to(device=device),
                                                                 requires_grad=True)


def to_tlm_array(pytorch_tlm: nn.Module, embeddings_keys: List[str] = None, name: str = "TLM") -> Dict:
    """Convert a Transformer LM-type module to a set of weights in a Dict

    :param pytorch_tlm: A Transformer LM-type module
    :param embeddings_keys: A key to get the embeddings from, defaults to `None` in which case, gets all keys
    :param name: A name for this TLM
    :return: A Dict containing all the keys to restore from Embeddings and the TransformerEncoderStack
    """
    d = {}
    transformer = pytorch_tlm.transformer if hasattr(pytorch_tlm, 'transformer') else pytorch_tlm.generator
    d.update(to_encoder_stack_array(transformer, name=f"{name}/TransformerEncoderStack"))
    keys_to_write = embeddings_keys if embeddings_keys else list(pytorch_tlm.embeddings.keys())

    for embeddings_key in keys_to_write:
        d.update(to_embed_array(pytorch_tlm.embeddings[embeddings_key], name=f"{name}/Embeddings/{embeddings_key}"))

    if hasattr(pytorch_tlm.embeddings.reduction, 'ln'):
        d.update(to_weight_array(pytorch_tlm.embeddings.reduction.ln, name=f"{name}/Embeddings/reduction/ln"))
    return d


def save_tlm_npz(pytorch_tlm: nn.Module, npz: str, embeddings_keys: List[str] = None, name: str = "TLM", verbose: bool = False):
    """Save a TLM to an NPZ file

    :param pytorch_tlm: A Transformer LM-type module
    :param npz: A file to save
    :param embeddings_keys: A key to get embeddings from.  Defaults to `None`, in which case, all embeddings are written
    :param name: A name for this TLM
    :param verbose: whether output 
    :return: None
    """
    d = to_tlm_array(pytorch_tlm, embeddings_keys, name)
    if verbose:
        print(d.keys())
    np.savez(npz, **d)


def to_encoder_stack_array(
    pytorch_encoder_stack: TransformerEncoderStack, name: str = "TransformerEncoderStack"
) -> Dict:
    """Convert a `TransformerEncoderStack` to a set of weigths

    :param pytorch_encoder_stack: A transformer encoder stack
    :param name: A name
    :return: A Dict containing a set of weights
    """
    d = {}
    if isinstance(pytorch_encoder_stack.ln, nn.LayerNorm):
        d.update(to_weight_array(pytorch_encoder_stack.ln, f"{name}/ln"))
    for i, enc_pyt in enumerate(pytorch_encoder_stack.encoders):
        d.update(to_encoder_array(enc_pyt, f"{name}/{i}"))
    return d


def from_encoder_stack_array(
    pytorch_encoder_stack: TransformerEncoderStack, d: Dict, name: str = "TransformerEncoderStack"
):
    """Restore weights from a `TransformerEncoderStack`

    :param pytorch_encoder_stack: A transformer encoder stack
    :param d: A Dict containing sets of arrays
    :param name: A name for this primitive
    :return: None
    """
    if isinstance(pytorch_encoder_stack.ln, nn.LayerNorm):
        from_weight_array(pytorch_encoder_stack.ln, d, f"{name}/ln")
    for i, enc_pyt in enumerate(pytorch_encoder_stack.encoders):
        from_encoder_array(enc_pyt, d, f"{name}/{i}")


def from_tlm_array(pytorch_tlm: nn.Module, d: Dict, embeddings_keys: List[str] = None, name: str = "TLM"):
    """Restore a TLM-like model (possibly a `nn.Module` for fine-tuning)

    We just populate the `TransformerEncoderStack` and the embeddings from weights, all other values remain
    uninitialized

    :param pytorch_tlm: A TLM-like model
    :param d: A Dict of weights to restore for each layer
    :param embeddings_keys: Name of embeddings to restore, defaults to `None`, in which case all embeddings are restored
    :param name: A name for this primitive
    :return:
    """
    transformer = pytorch_tlm.transformer if hasattr(pytorch_tlm, 'transformer') else pytorch_tlm.generator
    from_encoder_stack_array(transformer, d, name=f"{name}/TransformerEncoderStack")
    keys_to_restore = embeddings_keys if embeddings_keys else list(pytorch_tlm.embeddings.keys())
    for embeddings_key in keys_to_restore:
        from_embed_array(pytorch_tlm.embeddings[embeddings_key], d, f"{name}/Embeddings/{embeddings_key}")
        if isinstance(pytorch_tlm.embeddings[embeddings_key], LearnedPositionalLookupTableEmbeddingsWithBias):
            tt = LookupTableEmbeddings(vsz=2, dsz=pytorch_tlm.embeddings.output_dim)
            from_embed_array(tt, d, f"{name}/Embeddings/tt")
            pytorch_tlm.embeddings[embeddings_key].bias = nn.Parameter(tt.embeddings.weight[0])
        else:
            from_embed_array(pytorch_tlm.embeddings[embeddings_key], d, f"{name}/Embeddings/{embeddings_key}")
    if hasattr(pytorch_tlm.embeddings.reduction, 'ln'):
        from_weight_array(pytorch_tlm.embeddings.reduction.ln, d, f"{name}/Embeddings/reduction/ln")



def load_tlm_npz(pytorch_tlm: nn.Module, npz: str, embeddings_keys: List[str] = None, name: str = "TLM"):
    """Restore a TLM-like model (possibly a `nn.Module` for fine-tuning

    We just populate the `TransformerEncoderStack` and the embeddings from weights, all other values remain
    uninitialized

    :param pytorch_tlm: A TLM-like model
    :param npz: A file to restore the weights from
    :param embeddings_key: Name of embeddings to restore, defaults to `None` in which case we restore all embeddings
    :param name: A name for this primitive
    :return:
    """
    d = np.load(npz)
    from_tlm_array(pytorch_tlm, d, embeddings_keys, name)


def load_tlm_transformers_bin(pytorch_tlm: nn.Module, bin_file: str, replace_layers=BERT_HF_LAYER_MAP, replace_embeds=BERT_HF_EMBED_MAP):
    """For BERT transformer from HuggingFace, we need a TLM with EmbeddingsStack with 2 features and LN reduce

    The Transformer architecture used by BERT mirrors T2T (with layer norms coming after each transformer layer and
    a layer norm immediately following a sum reduce of learned-positional embeddings, the word LUT embeddings and
    a token-type embedding.  For many cases, the token-type embeddings is uninteresting, and should be simply set
    to 0.  For some cases, setting the token-type is critical.  In MEAD, to support the token type we need a
    `LookupTableEmbeddings`, and for the token itself, we need a `LearnedPositionalEmbedding` (which adds the two
    features of the token LUT and the position LUT together.  MEAD composes multiple features with the
    `EmbeddingsStack` primitive, and provides a reduction operator.  This is equivalent to BERT if we supply
    the `EmbeddingsStack` with 2 features and the `sum-layer-norm` reduction.  To do this, 2 vectorizers must be supplied
    to MEAD, one for the usual wordpiece tokenizers, and another tokenizer which has to produce the exact same number
    of tokens but with a token type value.  For example, for sentence-based token type delimiting, we would want to
    have a vectorizer that starts with 0, and every time it sees a `[SEP]` in the token itself, produces an incremented
    value, e.g. 0.... 1..... 2....

    For applications where the token-type is uninteresting, we can actually make a more efficient representation,
    which we do by creating a single `LearnedPositionalLookupTableEmbeddingsWithBias` embedding object as the single
    feature to the `EmbeddingsStack`.  This operation looks like a `LearnedPositionalLookupTableEmbeddings` object
    but it has a learnable bias parameter which is initialized to LUT index 0 of the BERT pretrained checkpoint.
    Additionally, because this object's goal is to replace the compositional approach of having 2 features, it also has
    a LayerNorm operator under the hood.  If you use this object, under normal circumstances, you do not need to bother
    providing `EmbeddingsStack` with any reduction as there is only 1 embedding.

    We want to support both types of models being passed in here by the user.  The way we do this, is to check the
    number of features in the `EmbeddingsStack`.  If its just 1, we can just monkey-patch the object as though it was
    the 2-feature equivalent, and after its loaded, restore the single feature so that when it goes back to the user
    they have what the put in originally

    :param pytorch_tlm: A Transformer LM
    :param bin_file: A HuggingFace PyTorch BERT checkpoint
    :param replace_layers: The mapping from HuggingFace Transformer keys to MEAD Transformer keys
    :param replace_embeds: The mapping from HuggingFace Embeddings key to MEAD Embeddings Keys
    :return:
    """
    d = torch.load(bin_file)
    transformer = pytorch_tlm.transformer if hasattr(pytorch_tlm, 'transformer') else pytorch_tlm.generator
    num_layers = len(transformer.encoders)
    mapped_keys = convert_transformers_keys(num_layers, d, replace_layers, replace_embeds)
    old_embeddings_stack = None
    k_0 = pytorch_tlm.embeddings.keys()[0]

    # There are 2 options to consider here
    # Option 1: the user doesnt care about token type embeddings, which means that the embedding type will just be
    #   the usual LP embeddings with an added bias term set to weight 0
    # Option 2: the user does care about token types and has provided a token type feature (presumed to be in the
    #   second key of the embeddings stack

    if isinstance(pytorch_tlm.embeddings[k_0], LearnedPositionalLookupTableEmbeddingsWithBias):
        old_embeddings_stack = pytorch_tlm.embeddings
        # we need to temporarily monkey patch the embeddings to load them, and then we can reset them to what they were
        d = {k_0: pytorch_tlm.embeddings[k_0], 'tt':  LookupTableEmbeddings(vsz=2, dsz=old_embeddings_stack.output_dim)}
        pytorch_tlm.embeddings = EmbeddingsStack(d, reduction='sum-layer-norm')
    unknown_keys = pytorch_tlm.load_state_dict(mapped_keys, strict=False)
    missing_keys = [key for key in unknown_keys.missing_keys if key != 'embeddings.embeddings.0.bias']

    if old_embeddings_stack:
        old_embeddings_stack[k_0].bias = nn.Parameter(pytorch_tlm.embeddings['tt'].embeddings.weight[0])
        old_embeddings_stack.reduction.ln = pytorch_tlm.embeddings.reduction.ln
        pytorch_tlm.embeddings = old_embeddings_stack

    return {'missing': missing_keys, 'unexpected': unknown_keys.unexpected_keys}
