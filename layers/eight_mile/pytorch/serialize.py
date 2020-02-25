import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from eight_mile.pytorch.layers import Dense, TransformerEncoderStack, TransformerEncoder, MultiHeadedAttention


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


def to_mha_array(pytorch_mha: MultiHeadedAttention, name: str) -> Dict:
    """Convert a `MultiHeadedAttention` module to a set of arrays

    :param pytorch_mha: A `MultiHeadedAttention` module for Transformers
    :param name: The name of the layer
    :return: A Dict containing the arrays by key
    """
    d = {}

    d.update(to_weight_array(pytorch_mha.w_Q, f"{name}/w_Q"))
    d.update(to_weight_array(pytorch_mha.w_K, f"{name}/w_K"))
    d.update(to_weight_array(pytorch_mha.w_V, f"{name}/w_V"))
    d.update(to_weight_array(pytorch_mha.w_O, f"{name}/w_O"))
    return d


def from_mha_array(pytorch_mha: MultiHeadedAttention, d: Dict, name: str):
    """Restore a `MultiHeadedAttention` module from a set of keys

    :param pytorch_mha: A `MultiHeadedAttention` module for Transformers
    :param d: A Dict of arrays by key
    :param name: The name of the layer
    """
    from_weight_array(pytorch_mha.w_Q, d, f"{name}/w_Q")
    from_weight_array(pytorch_mha.w_K, d, f"{name}/w_K")
    from_weight_array(pytorch_mha.w_V, d, f"{name}/w_V")
    from_weight_array(pytorch_mha.w_O, d, f"{name}/w_O")


def to_encoder_array(pytorch_encoder: TransformerEncoder, name: str) -> Dict:
    """Convert a `TransformerEncoder` layer to an set of numpy arrays

    :param pytorch_encoder: A `TransformerEncoder` layer
    :param name: The layer name
    :return: A Dict of arrays by key
    """
    d = {}
    d.update(to_weight_array(pytorch_encoder.ln1, f"{name}/ln1"))
    d.update(to_weight_array(pytorch_encoder.ln2, f"{name}/ln2"))
    d.update(to_mha_array(pytorch_encoder.self_attn, f"{name}/mha"))
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
    from_mha_array(pytorch_encoder.self_attn, d, f"{name}/mha")
    from_ffn_array(pytorch_encoder.ffn, d, f"{name}/ffn")


def to_embed_array(pytorch_embed: nn.Module, name: str) -> Dict:
    """Convert a simple lookup table embedding to a `weights` array

    :param pytorch_embed: An embedding module
    :param name: A layer name
    :return: A Dict containing the embedding `weights`
    """
    weights = pytorch_embed.weight.cpu().detach().numpy()
    return {f"{name}/weights": weights}


def from_embed_array(pytorch_embed: nn.Module, d: Dict, name: str):
    """Restore a simple lookup table embedding from a `weights` array

    :param pytorch_embed: An embedding module
    :param d: A Dict containing a `weights` array to restore
    :param name: name of the layer
    :return: None
    """
    pytorch_embed.weight = torch.nn.Parameter(torch.from_numpy(d[f"{name}/weights"]), requires_grad=True)


def to_tlm_array(pytorch_tlm: nn.Module, embeddings_key: str = 'x', name: str = "TLM") -> Dict:
    """Convert a Transformer LM-type module to a set of weights in a Dict

    :param pytorch_tlm: A Transformer LM-type module
    :param embeddings_key: A key to get the embeddings from (defaults to `x`)
    :param name: A name for this TLM
    :return: A Dict containing all the keys to restore from Embeddings and the TransformerEncoderStack
    """
    d = {}
    d.update(to_encoder_stack_array(pytorch_tlm.transformer, name=f"{name}/TransformerEncoderStack"))
    embedding_layer = pytorch_tlm.embeddings[embeddings_key]
    # Save the word embedding with name LookupTableEmbeddings. If the positional embedding is learned, save it with the
    # name PositionalEmbeddings
    d.update(to_embed_array(embedding_layer.embeddings, name=f"{name}/LookupTableEmbeddings"))
    if hasattr(embedding_layer, 'pos_embeddings'):
        d.update((to_embed_array(embedding_layer.pos_embeddings, name=f"{name}/PositionalEmbeddings")))
    return d


def save_tlm_npz(pytorch_tlm: nn.Module, npz: str, embeddings_key: str = 'x', name: str = "TLM"):
    """Save a TLM to an NPZ file

    :param pytorch_tlm: A Transformer LM-type module
    :param npz: A file to save
    :param embeddings_key: A key to get embeddings from (defaults to `x`)
    :param name: A name for this TLM
    :return: None
    """
    d = to_tlm_array(pytorch_tlm, embeddings_key, name)
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
    embedding_layer = pytorch_tlm.embeddings[embeddings_key]
    from_embed_array(embedding_layer.embeddings, d, f"{name}/LookupTableEmbeddings")
    if hasattr(embedding_layer, 'pos_embeddings'):
        from_embed_array(embedding_layer.pos_embeddings, d, f"{name}/PositionalEmbeddings")


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
