from keras.models import Model
from keras.layers import Dense, Convolution1D, Embedding, Input, merge, GlobalMaxPooling1D, Dropout


# Use the functional API since we support parallel convolutions
def create_model(embeddings, nc, filtsz, cmotsz, hsz, maxlen, pdrop, finetune):
    x = Input(shape=(maxlen,), dtype='int32', name='input')

    vocab_size = embeddings.weights.shape[0]
    embedding_dim = embeddings.dsz
    
    lut = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embeddings.weights], input_length=maxlen, trainable=finetune)
    
    embed = lut(x)

    mots = []
    for i, fsz in enumerate(filtsz):
        conv = Convolution1D(cmotsz, fsz, activation='relu', input_length=maxlen)(embed)
        gmp = GlobalMaxPooling1D()(conv)
        mots.append(gmp)

    joined = merge(mots, mode='concat')
    cmotsz_all = cmotsz * len(filtsz)
    drop1 = Dropout(pdrop)(joined)

    input_dim = cmotsz_all
    last_layer = drop1

    if hsz > 0:
        proj = Dense(hsz, input_dim=cmotsz_all, activation='relu')(drop1)
        drop2 = Dropout(pdrop)(proj)
        input_dim = hsz
        last_layer = drop2

    dense = Dense(output_dim=nc, input_dim=input_dim, activation='softmax')(last_layer)
    model = Model(input=[x], output=[dense])
    return model
