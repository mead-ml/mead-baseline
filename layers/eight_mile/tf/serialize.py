import numpy as np
from typing import Dict, List
from eight_mile.tf.layers import TransformerEncoderStack, TransformerEncoder, MultiHeadedAttention, FFN
from eight_mile.tf.embeddings import LookupTableEmbeddings, LearnedPositionalLookupTableEmbeddingsWithBias, LearnedPositionalLookupTableEmbeddings

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


def from_attn_array(tf_attn: tf.keras.layers.Layer, d: Dict, name: str):
    """Restore a self-attention module from a set of keys

    :param tf_attn: A self-attention module for Transformers, could be MultiHeadedAttention or
    MultiHeadedRelativeAttention
    :param d: A Dict of arrays by key
    :param name: The name of the layer
    """
    from_weight_array(tf_attn.w_Q, d, f"{name}/w_Q")
    from_weight_array(tf_attn.w_K, d, f"{name}/w_K")
    from_weight_array(tf_attn.w_V, d, f"{name}/w_V")
    from_weight_array(tf_attn.w_O, d, f"{name}/w_O")

    if hasattr(tf_attn, 'rpr_key'):
        tf_attn.rpr_key.set_weights([d[f"{name}/rpr_key"]])
        tf_attn.rpr_value.set_weights([d[f"{name}/rpr_value"]])


def from_encoder_array(tf_encoder: TransformerEncoder, d: Dict, name: str):
    """Restore a `TransformerEncoder` layer from a set of numpy arrays

    :param tf_encoder: A `TransformerEncoder` layer
    :param d: A Dict of arrays by key
    :param name: The layer name
    :return: None
    """
    from_weight_array(tf_encoder.ln1, d, f"{name}/ln1")
    from_weight_array(tf_encoder.ln2, d, f"{name}/ln2")
    from_attn_array(tf_encoder.self_attn, d, f"{name}/attn")
    from_ffn_array(tf_encoder.ffn, d, f"{name}/ffn")


def from_embed_array(tf_embed: tf.keras.layers.Layer, d: Dict, name: str, bias=None):
    """Restore a simple lookup table embedding from a `weights` array

    :param tf_embed: An embedding module
    :param d: A Dict containing a `weights` array to restore
    :param name: name of the layer
    :return: None
    """
    weights = [d[f"{name}/weights"]]
    if hasattr(tf_embed, 'pos'):
        pos_weights = d[f"{name}/pos_weights"]
        weights = [pos_weights] + weights
        if hasattr(tf_embed, 'bias') and bias is not None:
            weights = weights + [bias.reshape(1, -1)]

    tf_embed.set_weights(weights)


def from_encoder_stack_array(tf_encoder_stack: TransformerEncoderStack, d: Dict, name: str = "TransformerEncoderStack"):
    """Restore weights from a `TransformerEncoderStack`

    :param tf_encoder_stack: A transformer encoder stack
    :param d: A Dict containing sets of arrays
    :param name: A name for this primitive
    :return: None
    """
    if isinstance(tf_encoder_stack.ln, tf.keras.layers.LayerNormalization):
        from_weight_array(tf_encoder_stack.ln, d, f"{name}/ln")
    for i, enc_tf in enumerate(tf_encoder_stack.encoders):
        from_encoder_array(enc_tf, d, f"{name}/{i}")


def from_tlm_array(tf_tlm: tf.keras.layers.Layer, d: Dict, embeddings_keys: List[str] = None, name: str = "TLM"):
    """Restore a TLM-like model (possibly a `Model` for fine-tuning)

    We just populate the `TransformerEncoderStack` and the embeddings from weights, all other values remain
    uninitialized

    :param tf_tlm: A TLM-like model
    :param d: A Dict of weights to restore for each layer
    :param embeddings_keys: Name of the embeddings to restore, defaults to `None` in which case restore all embeddings
    :param name: A name for this primitive
    :return:
    """
    transformer = tf_tlm.transformer if hasattr(tf_tlm, 'transformer') else tf_tlm.generator
    from_encoder_stack_array(transformer, d, name=f"{name}/TransformerEncoderStack")

    keys_to_restore = embeddings_keys if embeddings_keys else list(tf_tlm.embeddings.keys())
    for embeddings_key in keys_to_restore:
        # If we get this class in we have to monkey patch the embeddings so we can add the bias
        if isinstance(tf_tlm.embeddings[embeddings_key], LearnedPositionalLookupTableEmbeddingsWithBias):
            bias = d[f"{name}/Embeddings/tt/weights"][0]
            from_embed_array(tf_tlm.embeddings[embeddings_key], d, f"{name}/Embeddings/{embeddings_key}", bias)
        else:
            from_embed_array(tf_tlm.embeddings[embeddings_key], d, f"{name}/Embeddings/{embeddings_key}")
    if hasattr(tf_tlm.embeddings.reduction, 'ln'):
        from_weight_array(tf_tlm.embeddings.reduction.ln, d, f"{name}/Embeddings/reduction/ln")


def load_tlm_npz(tf_tlm: tf.keras.layers.Layer, npz: str, embeddings_key: str = 'x', name: str = "TLM"):
    """Restore a TLM-like model (possibly a `Model` for fine-tuning

    We just populate the `TransformerEncoderStack` and the embeddings from weights, all other values remain
    uninitialized

    :param tf_tlm: A TLM-like model
    :param npz: A file to restore the weights from
    :param embeddings_key: The name of the embeddings to restore, defaults to `x`
    :param name: A name for this primitive
    :return:
    """
    d = np.load(npz)
    from_tlm_array(tf_tlm, d, embeddings_key, name)
