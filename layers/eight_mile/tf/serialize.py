import numpy as np
from typing import Dict
from eight_mile.tf.layers import TransformerEncoderStack, TransformerEncoder, MultiHeadedAttention, FFN
import tensorflow as tf


def from_weight_array(tf_layer: tf.keras.layers.Layer, d: Dict, name: str):
    """Read in {`LayerNorm`, `Linear`, `layers.Dense`} from `weights` and `bias` fields

    :param tf_layer: A layer to get weights for
    :param d: A Dict containing the arrays by key
    :param name: The name of this layer
    :return: None
    """
    weights = d[f"{name}/weights"]
    bias = d[f"{name}/bias"]
    tf_layer.set_weights([weights.T, bias])


def from_ffn_array(tf_ffn: FFN, d: Dict, name: str):
    """Restore an `FFN` layer's weights from a set of arrays

    :param tf_ffn: An `FFN` layer for Transformers
    :param d: A Dict containing the arrays by key
    :param name: The name of the layer
    :return: None
    """
    from_weight_array(tf_ffn.expansion, d, f"{name}/expansion")
    from_weight_array(tf_ffn.squeeze, d, f"{name}/squeeze")


def from_mha_array(tf_mha: MultiHeadedAttention, d: Dict, name: str):
    """Restore a `MultiHeadedAttention` module from a set of keys

    :param tf_mha: A `MultiHeadedAttention` module for Transformers
    :param d: A Dict of arrays by key
    :param name: The name of the layer
    """
    from_weight_array(tf_mha.w_Q, d, f"{name}/w_Q")
    from_weight_array(tf_mha.w_K, d, f"{name}/w_K")
    from_weight_array(tf_mha.w_V, d, f"{name}/w_V")
    from_weight_array(tf_mha.w_O, d, f"{name}/w_O")


def from_encoder_array(tf_encoder: TransformerEncoder, d: Dict, name: str):
    """Restore a `TransformerEncoder` layer from a set of numpy arrays

    :param tf_encoder: A `TransformerEncoder` layer
    :param d: A Dict of arrays by key
    :param name: The layer name
    :return: None
    """
    from_weight_array(tf_encoder.ln1, d, f"{name}/ln1")
    from_weight_array(tf_encoder.ln2, d, f"{name}/ln2")
    from_mha_array(tf_encoder.self_attn, d, f"{name}/mha")
    from_ffn_array(tf_encoder.ffn, d, f"{name}/ffn")


def from_embed_array(tf_embed: tf.keras.layers.Layer, d: Dict, name: str):
    """Restore a simple lookup table embedding from a `weights` array

    :param pytorch_embed: An embedding module
    :param d: A Dict containing a `weights` array to restore
    :param name: name of the layer
    :return: None
    """
    weights = d[f"{name}/weights"]
    tf_embed.set_weights([weights])


def from_encoder_stack_array(tf_encoder_stack: TransformerEncoderStack, d: Dict, name: str = "TransformerEncoderStack"):
    """Restore weights from a `TransformerEncoderStack`

    :param pytorch_encoder_stack: A transformer encoder stack
    :param d: A Dict containing sets of arrays
    :param name: A name for this primitive
    :return: None
    """
    from_weight_array(tf_encoder_stack.ln, d, f"{name}/ln")
    for i, enc_tf in enumerate(tf_encoder_stack.encoders):
        from_encoder_array(enc_tf, d, f"{name}/{i}")


def from_tlm_array(tf_tlm: tf.keras.layers.Layer, d: Dict, embeddings_key: str = 'x', name: str = "TLM"):
    """Restore a TLM-like model (possibly a `Model` for fine-tuning)

    We just populate the `TransformerEncoderStack` and the embeddings from weights, all other values remain
    uninitialized

    :param tf_tlm: A TLM-like model
    :param d: A Dict of weights to restore for each layer
    :param embeddings_key: The name of the embeddings to restore, defaults to `x`
    :param name: A name for this primitive
    :return:
    """
    from_encoder_stack_array(tf_tlm.transformer, d, name=f"{name}/TransformerEncoderStack")
    from_embed_array(tf_tlm.embeddings.embeddings[embeddings_key], d, name=f"{name}/SinusoidalPositionalEmbeddings")


def load_tlm_npz(tf_tlm: tf.keras.layers.Layer, npz: str, embeddings_key: str = 'x', name: str = "TLM"):
    """Restore a TLM-like model (possibly a `Model` for fine-tuning

    We just populate the `TransformerEncoderStack` and the embeddings from weights, all other values remain
    uninitialized

    :param pytorch_tlm: A TLM-like model
    :param npz: A file to restore the weights from
    :param embeddings_key: The name of the embeddings to restore, defaults to `x`
    :param name: A name for this primitive
    :return:
    """
    d = np.load(npz)
    from_tlm_array(tf_tlm, d, embeddings_key, name)
