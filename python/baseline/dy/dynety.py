import dynet as dy

def Linear(osz, isz, pc, name="Linear"):
    weight = pc.add_parameters((osz, isz), name="{}Weight".format(name))
    bias = pc.add_parameters(osz, name="{}Bias".format(name))

    def linear(input_: dy.Expression):
        output = weight * input_ + bias
        return output

    return linear

def LSTM(osz, isz, pc, layers=1):
    lstm = dy.VanillaLSTMBuilder(layers, isz, osz, pc)

    def encode(input_):
        state = lstm.initial_state()
        out = state.transduce(input_)
        return out[-1]

    return encode

def Convolution1d(fsz, cmotsz, dsz, pc, strides=(1, 1, 1, 1)):
    weight = pc.add_parameters((1, fsz, dsz, cmotsz), name='ConvWeight')
    bias = pc.add_parameters((cmotsz), name="ConvBias")

    def conv(input_):
        c = dy.conv2d_bias(input_, weight, bias, strides, is_valid=False)
        activation = dy.rectify(c)
        mot = dy.reshape(dy.max_dim(activation, 1), (cmotsz,))
        return mot

    return conv

def Embedding(
    vsz, dsz, pc,
    embedding_weight=None, finetune=False, dense=False, batched=True
):
    if embedding_weight is not None:
        embeddings = pc.lookup_parameters_from_numpy(embedding_weight, name="Embeddings")
    else:
        embeddings = pc.add_lookup_parameters((vsz, dsz), name="Embeddings")

    def embed(input_):
        lookup = dy.lookup_batch if batched else dy.lookup
        embedded = [lookup(embeddings, x, finetune) for x in input_]
        if dense:
            return dy.transpose(dy.concatenate(embedded, d=1))
        return embedded

    return embed
