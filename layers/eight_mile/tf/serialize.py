import numpy as np
from typing import Dict, List
from eight_mile.tf.layers import (
    TransformerEncoderStack,
    TransformerEncoder,
    TransformerDecoderStack,
    TransformerDecoder,
    MultiHeadedAttention,
    PassThru,
    FFN,
    SingleHeadReduction,
    SpatialGatingUnit,
    GatedMLPEncoder,
    GatedMLPEncoderStack
)
from eight_mile.tf.embeddings import LookupTableEmbeddings, LearnedPositionalLookupTableEmbeddingsWithBias, LearnedPositionalLookupTableEmbeddings

import tensorflow as tf


def to_weight_array(tf_layer: tf.keras.layers.Layer, name: str) -> Dict:
    """Convert a {`LayerNorm`, `tf.keras.layers.Dense`} to `weights` and `bias` arrays

    :param tf_layer: A layer to get weights for
    :param name: The name of this layer to serialize
    :return: A Dictionary containing `weights` and `bias` keys
    """

    weights, bias = tf_layer.get_weights()
    return {f"{name}/weights": weights.T, f"{name}/bias": bias}


def from_weight_array(tf_layer: tf.keras.layers.Layer, d: Dict, name: str):
    """Read in {`LayerNorm`, `tf.keras.layers.Dense`} from `weights` and `bias` fields

    :param tf_layer: A layer to get weights for
    :param d: A Dict containing the arrays by key
    :param name: The name of this layer
    :return: None
    """
    weights = d[f"{name}/weights"]
    bias = d[f"{name}/bias"]
    tf_layer.set_weights([weights.T, bias])


def to_ffn_array(tf_ffn: FFN, name: str) -> Dict:
    """Convert a `FFN` layer to a set of arrays

    :param tf_ffn: An `FFN` layer for Transformers
    :param name: The name of the layer
    :return: A Dict containing the arrays by key
    """
    d = {}
    d.update(to_weight_array(tf_ffn.expansion, f"{name}/expansion"))
    d.update(to_weight_array(tf_ffn.squeeze, f"{name}/squeeze"))
    return d


def from_ffn_array(tf_ffn: FFN, d: Dict, name: str):
    """Restore an `FFN` layer's weights from a set of arrays

    :param tf_ffn: An `FFN` layer for Transformers
    :param d: A Dict containing the arrays by key
    :param name: The name of the layer
    :return: None
    """
    from_weight_array(tf_ffn.expansion, d, f"{name}/expansion")
    from_weight_array(tf_ffn.squeeze, d, f"{name}/squeeze")


def to_attn_array(tf_attn: tf.keras.layers.Layer, name: str) -> Dict:
    """Convert a self-attention module to a set of arrays

    :param tf_attn: The self-attention layer of the transformer encoder, could be MultiHeadedAttention or
    MultiHeadedRelativeAttention
    :param name: The name of the layer
    :return: A Dict containing the arrays by key
    """
    d = {}

    d.update(to_weight_array(tf_attn.w_Q, f"{name}/w_Q"))
    d.update(to_weight_array(tf_attn.w_K, f"{name}/w_K"))
    d.update(to_weight_array(tf_attn.w_V, f"{name}/w_V"))

    if hasattr(tf_attn, 'w_O'):
        d.update(to_weight_array(tf_attn.w_O, f"{name}/w_O"))

    if hasattr(tf_attn, 'rpr_key'):
        # Embeddings have same shape in PyTorch and TF [input_sz, output_sz]
        rpr_key_weights = tf_attn.rpr_key.get_weights()[0]
        d.update({f"{name}/rpr_key": rpr_key_weights})

    if hasattr(tf_attn, 'rpr_value'):
        rpr_value_weights = tf_attn.rpr_value.get_weights()[0]
        d.update({f"{name}/rpr_value": rpr_value_weights})

    return d


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

    if hasattr(tf_attn, 'w_O'):
        from_weight_array(tf_attn.w_O, d, f"{name}/w_O")

    if hasattr(tf_attn, 'rpr_key'):
        tf_attn.rpr_key.set_weights([d[f"{name}/rpr_key"]])

    if hasattr(tf_attn, 'rpr_value'):
        tf_attn.rpr_value.set_weights([d[f"{name}/rpr_value"]])


def to_sgu_array(tf_sgu: SpatialGatingUnit, name: str) -> Dict:
    d = {}
    d.update(to_weight_array(tf_sgu.norm, f"{name}/norm"))
    d.update(to_weight_array(tf_sgu.proj, f"{name}/proj"))
    return d


def to_gmlp_encoder_array(tf_encoder: GatedMLPEncoder, name: str) -> Dict:
    """Convert a `GatedMLPEncoder` layer to an set of numpy arrays

    :param tf_encoder: A `GatedMLPEncoder` layer
    :param name: The layer name
    :return: A Dict of arrays by key
    """
    d = {}
    d.update(to_weight_array(tf_encoder.to_ffn, f"{name}/to_ffn"))
    d.update(to_weight_array(tf_encoder.from_sgu, f"{name}/from_sgu"))
    d.update(to_weight_array(tf_encoder.norm, f"{name}/norm"))
    d.update(to_sgu_array(tf_encoder.spatial_gating_unit, f"{name}/spatial_gating_unit"))
    return d

def to_encoder_array(tf_encoder: TransformerEncoder, name: str) -> Dict:
    """Convert a `TransformerEncoder` layer to an set of numpy arrays

    :param tf_encoder: A `TransformerEncoder` layer
    :param name: The layer name
    :return: A Dict of arrays by key
    """
    d = {}
    d.update(to_weight_array(tf_encoder.ln1, f"{name}/ln1"))
    d.update(to_weight_array(tf_encoder.ln2, f"{name}/ln2"))
    d.update(to_attn_array(tf_encoder.self_attn, f"{name}/attn"))
    d.update(to_ffn_array(tf_encoder.ffn, f"{name}/ffn"))
    return d


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


def to_decoder_array(tf_decoder: TransformerDecoder, name: str) -> Dict:
    """Convert a `TransformerDeccoder` layer to an set of numpy arrays

    :param tf_decoder: A `TransformerDecoder` layer
    :param name: The layer name
    :return: A Dict of arrays by key
    """
    d = {}
    d.update(to_weight_array(tf_decoder.ln1, f"{name}/ln1"))
    d.update(to_weight_array(tf_decoder.ln2, f"{name}/ln2"))
    d.update(to_weight_array(tf_decoder.ln3, f"{name}/ln3"))
    d.update(to_attn_array(tf_decoder.self_attn, f"{name}/self_attn"))
    d.update(to_attn_array(tf_decoder.src_attn, f"{name}/src_attn"))
    d.update(to_ffn_array(tf_decoder.ffn, f"{name}/ffn"))
    return d


def to_embed_array(tf_embed: tf.keras.layers.Layer, name: str) -> Dict:
    """Convert positional embedding to a `weights` array, if it's learned positional embedding,
    save the pos_weight as well.

    :param tf_embed: An embedding module
    :param name: A layer name
    :return: A Dict containing the embedding `weights`
    """
    d = {}


    if hasattr(tf_embed, 'pos'):
        pos_weights = tf.keras.backend.get_value(tf_embed.pos)
        d.update({f"{name}/pos_weights": pos_weights})

    if hasattr(tf_embed, 'bias'):
        bias = tf.keras.backend.get_value(tf_embed.bias)
        d.update({f"{name}/bias": bias.squeeze()})

    # Note: we override the Keras function forcing a single value not a list
    # this function could break if we used get_weights() since we have broken that
    # however, we'd like to fix that so using the raw value feels like the lesser of 2 evils
    weights = tf.keras.backend.get_value(tf_embed.W)
    d.update({f"{name}/weights": weights})
    return d


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


def to_gmlp_encoder_stack_array(
        tf_encoder_stack: GatedMLPEncoderStack, name: str = "GatedMLPEncoderStack"
) -> Dict:
    """Convert a `TransformerEncoderStack` to a set of weigths

    :param tf_encoder_stack: A transformer encoder stack
    :param name: A name
    :return: A Dict containing a set of weights
    """
    d = {}
    if isinstance(tf_encoder_stack.ln, tf.keras.layers.LayerNormalization):
        d.update(to_weight_array(tf_encoder_stack.ln, f"{name}/ln"))
    for i, enc_tf in enumerate(tf_encoder_stack.encoders):
        d.update(to_gmlp_encoder_array(enc_tf, f"{name}/{i}"))
    return d

def to_encoder_stack_array(
    tf_encoder_stack: TransformerEncoderStack, name: str = "TransformerEncoderStack"
) -> Dict:
    """Convert a `TransformerEncoderStack` to a set of weigths

    :param tf_encoder_stack: A transformer encoder stack
    :param name: A name
    :return: A Dict containing a set of weights
    """
    d = {}
    if isinstance(tf_encoder_stack.ln, tf.keras.layers.LayerNormalization):
        d.update(to_weight_array(tf_encoder_stack.ln, f"{name}/ln"))
    for i, enc_tf in enumerate(tf_encoder_stack.encoders):
        d.update(to_encoder_array(enc_tf, f"{name}/{i}"))
    return d



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



def to_decoder_stack_array(
    tf_decoder_stack: TransformerDecoderStack, name: str = "TransformerDecoderStack"
) -> Dict:
    """Convert a `TransformerEncoderStack` to a set of weights

    :param tf_encoder_stack: A transformer encoder stack
    :param name: A name
    :return: A Dict containing a set of weights
    """
    d = {}
    if isinstance(tf_decoder_stack.ln, tf.keras.layers.LayerNormalization):
        d.update(to_weight_array(tf_decoder_stack.ln, f"{name}/ln"))
    for i, dec_tf in enumerate(tf_decoder_stack.decoders):
        d.update(to_decoder_array(dec_tf, f"{name}/{i}"))
    return d

def from_decoder_array(tf_decoder: TransformerDecoder, d: Dict, name: str):
    """Restore a `TransformerDecoder` layer from a set of numpy arrays

    :param tf_decoder: A `TransformerDecoder` layer
    :param d: A Dict of arrays by key
    :param name: The layer name
    :return: None
    """
    from_weight_array(tf_decoder.ln1, d, f"{name}/ln1")
    from_weight_array(tf_decoder.ln2, d, f"{name}/ln2")
    from_weight_array(tf_decoder.ln3, d, f"{name}/ln3")
    from_attn_array(tf_decoder.src_attn, d, f"{name}/src_attn")
    from_attn_array(tf_decoder.self_attn, d, f"{name}/self_attn")
    from_ffn_array(tf_decoder.ffn, d, f"{name}/ffn")

def from_decoder_stack_array(
        tf_decoder_stack: TransformerDecoderStack, d: Dict, name: str = "TransformerDecoderStack"
):
    """Restore weights from a `TransformerDecoderStack`

    :param pytorch_decoder_stack: A transformer decoder stack
    :param d: A Dict containing sets of arrays
    :param name: A name for this primitive
    :return: None
    """
    if isinstance(tf_decoder_stack.ln, tf.keras.layers.LayerNormalization):
        from_weight_array(tf_decoder_stack.ln, d, f"{name}/ln")
    for i, dec_pyt in enumerate(tf_decoder_stack.decoders):
        from_decoder_array(dec_pyt, d, f"{name}/{i}")


def to_tlm_array(tf_tlm: tf.keras.layers.Layer, embeddings_keys: List[str] = None, name: str = "TLM") -> Dict:
    """Convert a Transformer LM-type module to a set of weights in a Dict

    :param pytorch_tlm: A Transformer LM-type module
    :param embeddings_keys: A key to get the embeddings from, defaults to `None` in which case, gets all keys
    :param name: A name for this TLM
    :return: A Dict containing all the keys to restore from Embeddings and the TransformerEncoderStack
    """
    d = {}
    transformer = tf_tlm.transformer if hasattr(tf_tlm, 'transformer') else tf_tlm.generator
    if isinstance(transformer, GatedMLPEncoderStack):
        d.update(to_gmlp_encoder_stack_array(transformer, name=f"{name}/GatedMLPEncoderStack"))
    else:
        d.update(to_encoder_stack_array(transformer, name=f"{name}/TransformerEncoderStack"))
    keys_to_write = embeddings_keys if embeddings_keys else list(tf_tlm.embeddings.keys())

    for embeddings_key in keys_to_write:
        d.update(to_embed_array(tf_tlm.embeddings[embeddings_key], name=f"{name}/Embeddings/{embeddings_key}"))

    if hasattr(tf_tlm.embeddings.reduction, 'ln'):
        d.update(to_weight_array(tf_tlm.embeddings.reduction.ln, name=f"{name}/Embeddings/reduction/ln"))
    return d


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


def save_tlm_npz(tf_tlm: tf.keras.layers.Layer, npz: str, embeddings_keys: List[str] = None, name: str = "TLM", verbose: bool = False):
    """Save a TLM to an NPZ file

    :param tf_tlm: A Transformer LM-type module
    :param npz: A file to save
    :param embeddings_keys: A key to get embeddings from.  Defaults to `None`, in which case, all embeddings are written
    :param name: A name for this TLM
    :param verbose: whether output
    :return: None
    """
    d = to_tlm_array(tf_tlm, embeddings_keys, name)
    if verbose:
        print(d.keys())
    np.savez(npz, **d)


def save_tlm_output_npz(tlm: tf.keras.layers.Layer, npz: str, embeddings_keys: List[str] = None, name: str = "TLM", verbose: bool = False):
    """Save a TLM to an NPZ file with an output layer

    :param tlm: A Transformer LM-type module
    :param npz: A file to save
    :param embeddings_keys: A key to get embeddings from.  Defaults to `None`, in which case, all embeddings are written
    :param name: A name for this TLM
    :param verbose: whether output
    :return: None
    """
    d = to_tlm_array(tlm, embeddings_keys, name)
    if hasattr(tlm, 'output'):
        d.update(to_weight_array(tlm.output, name=f'{name}/output'))
    elif hasattr(tlm, 'output_layer'):
        d.update(to_weight_array(tlm.output_layer, name=f'{name}/output'))
    if verbose:
        print(d.keys())
    np.savez(npz, **d)


def save_transformer_seq2seq_npz(tf_seq2seq: tf.keras.layers.Layer, npz: str, src_embeddings_keys: List[str] = None,
                                 tgt_embedding_key: str = 'y', name: str = "Seq2Seq", verbose: bool = False):
    """Save a Transformer seq2seq file out

    The will be in tf_seq2seq.encoder.transformer, and the usual conversions work for that (via `to_tlm_array()`).
    The decoder requires a new converter for the portion containing attention weights between the encoder and the decoder

    :param tf_seq2seq: A Transformer Seq2Seq module
    """
    #enc = to_tlm_array(tf_seq2seq.encoder, embeddings_keys, name=f'{name}/Encoder')

    enc = {}
    transformer = tf_seq2seq.encoder.transformer
    enc.update(to_encoder_stack_array(transformer, name=f"{name}/TransformerEncoderStack"))
    enc_keys_to_write = src_embeddings_keys if src_embeddings_keys else list(tf_seq2seq.src_embeddings.keys())

    for embeddings_key in enc_keys_to_write:
        enc.update(to_embed_array(tf_seq2seq.src_embeddings[embeddings_key], name=f"{name}/SrcEmbeddings/{embeddings_key}"))

    dec = {}
    transformer_decoder = tf_seq2seq.decoder.transformer_decoder

    dec.update(to_decoder_stack_array(transformer_decoder, name=f"{name}/TransformerDecoderStack"))
    dec.update(to_embed_array(tf_seq2seq.tgt_embedding, name=f"{name}/TgtEmbedding/{tgt_embedding_key}"))

    if verbose:
        print(enc.keys())
        print(dec.keys())
    np.savez(npz, **enc, **dec)


def load_transformer_seq2seq_npz(tf_seq2seq: tf.keras.layers.Layer,
                                 npz: str, src_embeddings_keys: List[str] = None,
                                 tgt_embedding_key: str = 'y', name: str = "Seq2Seq"):
    """Save a Transformer seq2seq file out

    The will be in pytorch_seq2seq.encoder.transformer, and the usual conversions work for that (via `to_tlm_array()`).
    The decoder requires a new converter for the portion containing attention weights between the encoder and the decoder

    :param pytorch_seq2seq: A Transformer Seq2Seq module
    :param npz: The file name
    :param src_embeddings_keys: An optional list of the src embeddings keys to load, otherwise use what we find
    :param tgt_embedding_key: An optional tgt embedding, otherwise assume 'y' (TODO: bad assumption?)
    :param name: An optional name of the model in the NPZ, otherwise assume `Seq2Seq`
    """

    d = np.load(npz)

    transformer = tf_seq2seq.encoder.transformer
    from_encoder_stack_array(transformer, d, name=f"{name}/TransformerEncoderStack")

    enc_keys_to_restore = src_embeddings_keys if src_embeddings_keys else list(tf_seq2seq.src_embeddings.keys())

    for embeddings_key in enc_keys_to_restore:
        from_embed_array(tf_seq2seq.src_embeddings[embeddings_key], d, f"{name}/SrcEmbeddings/{embeddings_key}")

    transformer_decoder = tf_seq2seq.decoder.transformer_decoder

    from_decoder_stack_array(transformer_decoder, d,  name=f"{name}/TransformerDecoderStack")
    from_embed_array(tf_seq2seq.decoder.tgt_embeddings, d, name=f"{name}/TgtEmbedding/{tgt_embedding_key}")


def to_attn_pool_array(tf_attn_pool: tf.keras.layers.Layer, name: str) -> Dict:
    """Convert a self-attention module to a set of arrays

    :param tf_attn_pool: The attention pooling layer of a dual encoder, this could be single head or 2-headed
    :return: A Dict containing the arrays by key
    """
    d = {}

    if isinstance(tf_attn_pool, SingleHeadReduction):
        d.update(to_weight_array(tf_attn_pool.w_Q, f"{name}/w_Q"))
        d.update(to_weight_array(tf_attn_pool.w_K, f"{name}/w_K"))
    else:
        d.update(to_weight_array(tf_attn_pool.reduction1.w_Q, f"{name}/reduction1/w_Q"))
        d.update(to_weight_array(tf_attn_pool.reduction1.w_K, f"{name}/reduction1/w_K"))

        d.update(to_weight_array(tf_attn_pool.reduction2.w_Q, f"{name}/reduction2/w_Q"))
        d.update(to_weight_array(tf_attn_pool.reduction2.w_K, f"{name}/reduction2/w_K"))
    return d


def from_attn_pool_array(tf_attn_pool: tf.keras.layers.Layer, d: Dict, name: str):
    """Restore a self-attention pooling module from a set of keys

    :param tf_attn_pool: The attention pooling layer of a dual encoder, this could be single head or 2-headed
    :param d: A Dict of arrays by key
    :param name: The name of the layer
    """

    if isinstance(tf_attn_pool, SingleHeadReduction):
        from_weight_array(tf_attn_pool.w_Q, d, f"{name}/w_Q")
        from_weight_array(tf_attn_pool.w_K, d, f"{name}/w_K")

    else:
        from_weight_array(tf_attn_pool.reduction1.w_Q, d, f"{name}/reduction1/w_Q")
        from_weight_array(tf_attn_pool.reduction1.w_K, d, f"{name}/reduction1/w_K")
        from_weight_array(tf_attn_pool.reduction2.w_Q, d, f"{name}/reduction2/w_Q")
        from_weight_array(tf_attn_pool.reduction2.w_K, d, f"{name}/reduction2/w_K")


def save_transformer_de_npz(tf_de: tf.keras.layers.Layer, npz: str, embeddings_keys: List[str] = None,
                            name: str = "TLM", verbose: bool = False):
    """Save a Transformer de file out

    A Dual-Encoder will have 2 transformer layers with shared weights.  Because of this, when we save we only
    want to save the first layer.  However, the upper "stacking" layers may be different

    The encoder will have a transformer stack followed optionally by single or dual headed attention, and finally
    either a linear or FFN stack of layers

    :param tf_de: A Transformer Dual Encoder module
    """

    enc = {}
    transformer = tf_de.transformer
    enc.update(to_encoder_stack_array(transformer, name=f"{name}/TransformerEncoderStack"))
    enc_keys_to_write = embeddings_keys if embeddings_keys else list(tf_de.embeddings.keys())

    for embeddings_key in enc_keys_to_write:
        enc.update(to_embed_array(tf_de.embeddings[embeddings_key], name=f"{name}/Embeddings/{embeddings_key}"))

    enc.update(to_attn_pool_array(tf_de.reduction_layer, f"{name}/ReductionLayer"))

    ff1 = tf_de.ff1
    if isinstance(ff1, tf.keras.layers.Dense):
        enc.update(to_weight_array(ff1, f"{name}/ff1"))
    elif not isinstance(ff1, PassThru):
        raise Exception("We dont currently support stacking layers in dual-encoder serialization")
    ff2 = tf_de.ff2
    if isinstance(ff2, tf.keras.layers.Dense):
        enc.update(to_weight_array(ff2, f"{name}/ff2"))
    elif not isinstance(ff2, PassThru):
        raise Exception("We dont currently support stacking layers in dual-encoder serialization")

    np.savez(npz, **enc)


def load_transformer_de_npz(tf_de: tf.keras.layers.Layer,
                            npz: str, embeddings_keys: List[str] = None,
                            name: str = "TLM"):
    """Load a dual-encoder from NPZ

    A Dual-Encoder will have 2 transformer layers with shared weights.  Because of this, when we save we only
    want to save the first layer.  However, the upper "stacking" layers may be different

    The encoder will have a transformer stack followed optionally by single or dual headed attention, and finally
    either a linear or FFN stack of layers

    :param tf_de: A Transformer Dual Encoder module
    :param npz:
    :param embeddings_keys:
    :param name:
    :return:
    """

    d = np.load(npz)

    transformer = tf_de.encoder.transformer
    from_encoder_stack_array(transformer, d, name=f"{name}/TransformerEncoderStack")

    enc_keys_to_restore = embeddings_keys if embeddings_keys else list(tf_de.embeddings.keys())

    for embeddings_key in enc_keys_to_restore:
        from_embed_array(tf_de.embeddings[embeddings_key], d, f"{name}/Embeddings/{embeddings_key}")

    from_attn_pool_array(tf_de.reduction, d, name=f"{name}/ReductionLayer")

    if hasattr(tf_de, 'ff1'):
        ff1 = tf_de.ff1
        if isinstance(ff1, tf.keras.layers.Dense):
            from_weight_array(ff1, d, f"{name}/ff1")
        elif not isinstance(ff1, PassThru):
            raise Exception("We dont currently support stacking layers in dual-encoder serialization")
    if hasattr(tf_de, 'ff2'):
        ff2 = tf_de.ff2
        if isinstance(ff2, tf.keras.layers.Dense):
            from_weight_array(ff2, d, f"{name}/ff2")
        elif not isinstance(ff2, PassThru):
            raise Exception("We dont currently support stacking layers in dual-encoder serialization")


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
