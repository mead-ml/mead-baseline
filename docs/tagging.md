# Structured Prediction using RNNs

This code is useful for tagging tasks, e.g., POS tagging, chunking and NER tagging.  Recently, several researchers have proposed using RNNs for tagging, particularly LSTMs.  These models do back-propagation through time (BPTT)
and then apply a shared fully connected layer per RNN output to produce a label.
A common modification is to use Bidirectional LSTMs -- one in the forward direction, and one in the backward direction.  This usually improves the resolution of the tagger significantly.  That is the default approach taken here.

To execute these models it is necessary to form word vectors.  It has been shown that character-level modeling is important in deep models to support morpho-syntatic structure for tagging tasks.
It seems that a good baseline should combine word vectors and character-level compositions of words.

## tag_char_rnn: word/character-based RNN tagger

The code uses word and character-level word embeddings.  For character-level processing, a character vector depth is selected, along with a word-vector depth. 

The character-level embeddings are based on Character-Aware Neural Language Models (Kim, Jernite, Sontag, Rush) and dos Santos 2014 (though the latter's tagging model is quite different).  Unlike dos Santos' approach, here parallel filters are applied during the convolution (which is like the Kim approach). Unlike the Kim approach residual connections of like size filters are used, and since they improve performance for tagging, word vectors are also used.

Twitter is a challenging data source for tagging problems.  The [TweetNLP project](http://www.cs.cmu.edu/~ark/TweetNLP) includes hand annotated POS data. The original taggers used for this task are described [here](http://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf).  The baseline that they compared their algorithm against got 83.38% accuracy.  The final model got 89.37% accuracy with many custom features.  Below, our simple BLSTM baseline with no custom features, and a very coarse approach to compositional character to word modeling still gets *88%-90%* accuracy.

### Pre-trained word2vec embeddings

#### Google News
The Google news word vectors are useful for many different tasks and tagging is no exception.  Here is a link to get those (from Google)

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

#### Twitter

Custom word2vec embeddings generated from ~32M tweets including s140 and the oct27 train+dev data.  Some of the data was sampled and preprocessed to have placeholder words for hashtags, mentions and URLs to be used as backoffs for words of those classes which are not found.

https://drive.google.com/drive/folders/0B8N1oYmGLVGWWWZYS2E0MlRXajQ?usp=sharing

If you use these, pass `--web_cleanup 1` to make sure that preprocessing is done to match preprocessing on the pre-trained embeddings

## Running It

Here is an example using convolutional filters for character embeddings, alongside word embeddings.  This is basically a combination of the dos Santos approach with the Kim parallel filter idea using TensorFlow:

```
python tag_char_rnn.py --rnntype blstm --optim sgd --wsz 30 --eta 0.01 \
    --epochs 40 --web_cleanup 1 --batchsz 20 --hsz 200 \
    --train ../data/oct27.train \
    --valid ../data/oct27.dev \
    --test ../data/oct27.test \
    --embed /data/xdata/oct-s140clean-uber.cbow-bin --cfiltsz 1 2 3 4 5 7
```

To run with PyTorch, just pass `--backend pytorch`

If you want to use only the convolutional filter word vectors (and no word embeddings), just remove the -embed line above.

### NER (and other IOB-type) Tagging

NER tagging can be performed with a BLSTM with the usage below:

```
python tag_char_rnn.py --rnntype blstm --patience 70 --layers 2 --optim sgd --eta 0.001 --epochs 1000 --batchsz 50 --hsz 100 \
    --train ../data/eng.train \
    --valid ../data/eng.testa \
    --test  ../data/eng.testb \
    --embed /data/xdata/GoogleNews-vectors-negative300.bin \
    --cfiltsz 1 2 3 4 5 7  --wsz 30
```

This will report an F1 score on at each validation pass, and will use F1 for early-stopping as well.

### Global coherency with a CRF (currently TensorFlow only)

For tasks that require global coherency like NER tagging, it has been shown that using a transition matrix between label states in conjunction with the output RNN tags improves performance.  This makes the tagger a linear chain CRF, and we can do this by simply adding another layer on top of our RNN output.  To do this, simply pass `--crf 1` as an argument.

```
python tag_char_rnn.py \
    --rnntype blstm --patience 70 \
    --layers 2 --optim sgd --eta 0.001 --epochs 1000 --batchsz 50 --hsz 100 \
    --train ../data/eng.train \
    --valid ../data/eng.testa \
    --test  ../data/eng.testb \
    --embed /data/xdata/GoogleNews-vectors-negative300.bin \
    --cfiltsz 1 2 3 4 5 7 --wsz 30 --crf 1
```

## Status

This model is implemented in TensorFlow and PyTorch (there is an old version in Torch7, which is no longer supported). The PyTorch CRF implementation is still experimental and in need of optimization.

_TODO_: Benchmark for CONLL NER

### Latest Runs

Here are the last observed performance scores using _tag_char_rnn_ with a 1-layer BLSTM on Twitter POS.  It was run for up to 40 epochs.

| Dataset   | Metric | Method    | Eta (LR) | Framework  | Score |
| --------- | ------ | --------- | -------  | ---------- | ----- |
| twpos-v03 |    Acc | SGD mom.  |     0.01 | TensorFlow | 89.7  |
| twpos-v03 |    Acc | Adam      |       -- | PyTorch    | 89.4  |

