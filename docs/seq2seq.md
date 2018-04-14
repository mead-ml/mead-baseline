# Seq2Seq

Encoder-decoder frameworks have been used for Statistical Machine Translation, Image Captioning, ASR, Conversation Modeling, Formula Generation and many other applications.  Seq2seq is a type of encoder-decoder implementation, where the encoder is some sort of RNN, and the memories (sometimes known as the "thought vector") are transferred over to the decoder, also an RNN, essentially making this a conditional language model.  The code as it is written here is with text in mind, and supports multiple types of RNNs, including GRUs and LSTMs, as well as stacked layer.  Global attention is optional.

## seq2seq: Phrase in, phrase-out, 2 lookup table implementation, input is a temporal vector, and so is output

This code implements seq2seq with mini-batching (as in other examples) using adagrad, adadelta, sgd or adam.  It supports two vocabularies, and supports word2vec pre-trained models as input, filling in words that are attested in the dataset but not found in the pre-trained models.  It uses dropout for regularization.

For any reasonable size data, this really needs to run on the GPU for realistic training times.

## Status

This model is implemented in TensorFlow and PyTorch.  The English-Vietnamese dataset is from https://nlp.stanford.edu/projects/nmt/ and the site authors have published previously on the dataset here: https://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf. Test results are for Single NMT on TED tst2013.  The WMT dataset is available via https://github.com/tensorflow/nmt#wmt-german-english


*Our results for Single NMT*

| dataset        | metric | optim  | eta (LR) | backend    | score  | encoder | layers | dropout | hidden | embed | epochs |
| -------------- | ------ | ------ | -------- | ---------- | ------ | ------- | ------ | ------- | ------ | ----- | ------ |
| iwslt15-en-vi  |  BLEU  | adam   |  0.001   | TensorFlow | 25.21  | blstm   |      2 |     0.5 |   512  |  512  |    16  |
| newstest2015.(de|en) | BLEU | adam | 0.001  | TensorFlow | 27.92  | blstm   |      4 |     0.5 |   512  |  512  |    12  |

