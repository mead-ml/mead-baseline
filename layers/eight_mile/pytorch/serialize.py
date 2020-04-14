import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from eight_mile.pytorch.layers import Dense, TransformerEncoderStack, TransformerEncoder, MultiHeadedAttention


# BERT HuggingFace Tokenizers checkpoints can be converted into MEAD Baseline Transformer checkpoints
# With a simple name change
BERT_HF_LAYER_MAP = {
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
    'bert.embeddings.token_type_embeddings.weight': 'embeddings.embeddings.0.tok_embeddings.weight',
    'bert.embeddings.LayerNorm.beta': 'embeddings.embeddings.0.ln.bias',
    'bert.embeddings.LayerNorm.gamma': 'embeddings.embeddings.0.ln.weight',
}


def convert_transformers_keys(num_layers: int, d: Dict, replace_layer_map: Dict, replace_embed_map: Dict) -> Dict:
    m = {}
    for i in range(num_layers):
        for k, v in replace_layer_map.items():
            m[v.format(i)] = d[k.format(i)]

    for k, v in replace_embed_map.items():
        m[v] = d[k]

    return m


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
    pytorch_layer.weight = nn.Parameter(torch.from_numpy(d[f"{name}/weights"]), requires_grad=True)
    pytorch_layer.bias = nn.Parameter(torch.from_numpy(d[f"{name}/bias"]), requires_grad=True)


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
        pytorch_attn.rpr_key.weight = torch.nn.Parameter(torch.from_numpy(d[f"{name}/rpr_key"]))
        pytorch_attn.rpr_value.weight = torch.nn.Parameter(torch.from_numpy(d[f"{name}/rpr_value"]))


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
    pytorch_embed.embeddings.weight = torch.nn.Parameter(torch.from_numpy(d[f"{name}/weights"]), requires_grad=True)
    if hasattr(pytorch_embed, 'pos_embeddings'):
        pytorch_embed.pos_embeddings.weight = torch.nn.Parameter(torch.from_numpy(d[f"{name}/pos_weights"]),
                                                                 requires_grad=True)


def to_tlm_array(pytorch_tlm: nn.Module, embeddings_key: str = 'x', name: str = "TLM") -> Dict:
    """Convert a Transformer LM-type module to a set of weights in a Dict

    :param pytorch_tlm: A Transformer LM-type module
    :param embeddings_key: A key to get the embeddings from (defaults to `x`)
    :param name: A name for this TLM
    :return: A Dict containing all the keys to restore from Embeddings and the TransformerEncoderStack
    """
    d = {}
    d.update(to_encoder_stack_array(pytorch_tlm.transformer, name=f"{name}/TransformerEncoderStack"))
    d.update(to_embed_array(pytorch_tlm.embeddings[embeddings_key], name=f"{name}/PositionalEmbeddings"))
    return d


def save_tlm_npz(pytorch_tlm: nn.Module, npz: str, embeddings_key: str = 'x', name: str = "TLM", verbose: bool = False):
    """Save a TLM to an NPZ file

    :param pytorch_tlm: A Transformer LM-type module
    :param npz: A file to save
    :param embeddings_key: A key to get embeddings from (defaults to `x`)
    :param name: A name for this TLM
    :param verbose: whether output 
    :return: None
    """
    d = to_tlm_array(pytorch_tlm, embeddings_key, name)
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
    from_weight_array(pytorch_encoder_stack.ln, d, f"{name}/ln")
    for i, enc_pyt in enumerate(pytorch_encoder_stack.encoders):
        from_encoder_array(enc_pyt, d, f"{name}/{i}")


def from_tlm_array(pytorch_tlm: nn.Module, d: Dict, embeddings_key: str = 'x', name: str = "TLM"):
    """Restore a TLM-like model (possibly a `nn.Module` for fine-tuning)

    We just populate the `TransformerEncoderStack` and the embeddings from weights, all other values remain
    uninitialized

    :param pytorch_tlm: A TLM-like model
    :param d: A Dict of weights to restore for each layer
    :param embeddings_key: The name of the embeddings to restore, defaults to `x`
    :param name: A name for this primitive
    :return:
    """
    from_encoder_stack_array(pytorch_tlm.transformer, d, name=f"{name}/TransformerEncoderStack")
    from_embed_array(pytorch_tlm.embeddings[embeddings_key], d, f"{name}/PositionalEmbeddings")


def load_tlm_npz(pytorch_tlm: nn.Module, npz: str, embeddings_key: str = 'x', name: str = "TLM"):
    """Restore a TLM-like model (possibly a `nn.Module` for fine-tuning

    We just populate the `TransformerEncoderStack` and the embeddings from weights, all other values remain
    uninitialized

    :param pytorch_tlm: A TLM-like model
    :param npz: A file to restore the weights from
    :param embeddings_key: The name of the embeddings to restore, defaults to `x`
    :param name: A name for this primitive
    :return:
    """
    d = np.load(npz)
    from_tlm_array(pytorch_tlm, d, embeddings_key, name)

def load_tlm_transformers_bin(pytorch_tlm: nn.Module, bin_file: str, replace_layers=BERT_HF_LAYER_MAP, replace_embeds=BERT_HF_EMBED_MAP):
    d = torch.load(bin_file)
    num_layers = len(pytorch_tlm.transformer.encoders)
    mapped_keys = convert_transformers_keys(num_layers, d, replace_layers, replace_embeds)
    unknown_keys = pytorch_tlm.load_state_dict(mapped_keys, strict=False)
    print('Ignored ', unknown_keys)