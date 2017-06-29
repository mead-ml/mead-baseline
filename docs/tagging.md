# Structured Prediction using RNNs

This code is useful for tagging tasks, e.g., POS tagging, chunking and NER tagging.  Recently, several researchers have proposed using RNNs for tagging, particularly LSTMs.  These models do back-propagation through time (BPTT)
and then apply a shared fully connected layer per RNN output to produce a label.
A common modification is to use Bidirectional LSTMs -- one in the forward direction, and one in the backward direction.  This usually improves the resolution of the tagger significantly.  That is the default approach taken here.

To execute these models it is necessary to form word vectors.  It has been shown that character-level modeling is important in deep models to support morpho-syntatic structure for tagging tasks.
It seems that a good baseline should combine word vectors and character-level compositions of words.

## tag_char_rnn: word/character-based RNN tagger

The code is fairly flexible and supports word and/or character-level word embeddings.  For character-level processing, a character vector depth is selected, along with a word-vector depth. There are two implementations of word vectors from character vectors: continous bag of letters or convolutional nets. In the latter case, the user specifies 1 or more parallel filter sizes, and the convolutional layer outputs the requested number of feature maps over time. The max value over time is selected for each, resulting in a fixed width word vector. This is essentially Kim's 2015 language model approach to forming word vectors. The other, simpler way, is just a sum over the character vectors for each word. This actually still works quite well on some datasets, as long as this character word representation is used alongside actual word vectors. In this case, the character vector size and word-char vector size are forced to be the same.

Word-based word vectors are supported by passing an --embed option to a word2vec file.  If this file is not passed, word-based word embeddings are not used at all.  If it is passed, word-char embeddings are concatenated to these representations. This is essentially the dos Santos 2014 approach to building vectors -- though the overall tagger is quite different, as it employs BLSTMs instead for the actual labeling.

Twitter is a challenging data source for tagging problems.  The [TweetNLP project](http://www.cs.cmu.edu/~ark/TweetNLP) includes hand annotated POS data. The original taggers used for this task are described [here](http://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf).  The baseline that they compared their algorithm against got 83.38% accuracy.  The final model got 89.37% accuracy with many custom features.  Below, our simple BLSTM baseline with no custom features, and a very coarse approach to compositional character to word modeling still gets *88%-90%* accuracy.

This has been tested on oct27 train, dev and test splits (http://www.cs.cmu.edu/~ark/TweetNLP), using custom word2vec embeddings generated from ~32M tweets including s140 and the oct27 train+dev data (download here: https://drive.google.com/drive/folders/0B8N1oYmGLVGWWWZYS2E0MlRXajQ?usp=sharing).  Some of the data was sampled and preprocessed to have placeholder words for hashtags, mentions and URLs to be used as backoffs for words of those classes which are not found.  The example below employs character vectors taken from splitting oct27 train+dev and s140 data and uses them to summed word vectors over characters.  Note that passing -cembed is not necessary, but provides a warm start for the character embeddings.

## Running It

Here is an example using convolutional filters for character embeddings, alongside word embeddings.  This is basically a combination of the dos Santos approach with the Kim parallel filter idea using TensorFlow:

```
python tag_char_rnn.py --epochs 40 --train $OCT_SPLITS/oct27.train \
    --valid $OCT_SPLITS/oct27.dev --test $OCT_SPLITS/oct27.test \
    --embed /data/xdata/oct-s140clean-uber.cbow-bin \
    --cfiltsz 1 2 3 4 5 7
```

If you want to use only the convolutional filter word vectors (and no word embeddings), just remove the -embed line above.

### NER (and other IOB-type) Tagging

To do NER tagging, we typically do not want to use accuracy as a metric.  In those cases, most often, we use F1, the harmonic mean of precision and recall.  If you have IOB tagged data,
you will typically want to pass '--fscore 1' to the code like so:

```
python tag_char_rnn.py --rnn blstm --patience 70 --numrnn 2 \
   --eta 0.001 --epochs 600 --batchsz 50 --hsz 100 \
   --train $CONLL/eng.train \
   --valid $CONLL/eng.testa \
   --test  $CONLL/eng.testb \
   --embed /data/xdata/GoogleNews-vectors-negative300.bin \
   --cfiltsz 1 2 3 4 5 7 --fscore 1
```

This will report an F1 score on at each validation pass, and will use F1 for early-stopping as well.

### Global coherency with a CRF (currently TensorFlow only)

For tasks that require global coherency like NER tagging, it has been shown that using a transition matrix between label states in conjunction with the output RNN tags improves performance.  This makes the tagger a linear chain CRF, and we can do this by simply adding another layer on top of our RNN output.  To do this, simply pass --crf as an argument.

```
python tag_char_rnn.py --rnn blstm --patience 70 --numrnn 2 \
   --eta 0.001 --epochs 600 --batchsz 50 --hsz 100 \
   --train $CONLL/eng.train \
   --valid $CONLL/eng.testa \
   --test  $CONLL/eng.testb \
   --embed /data/xdata/GoogleNews-vectors-negative300.bin \
   --cfiltsz 1 2 3 4 5 7 --fscore 1 --crf
```

## Status

This model is implemented in TensorFlow and PyTorch (there is an old version in Torch7, which is no longer supported).  The TensorFlow currently is the only implementation that supports using a CRF layer on the top.

_TODO_: Benchmark for CONLL NER

### Latest Runs

Here are the last observed performance scores using _tag_char_rnn_ with a 1-layer BLSTM on Twitter POS.  It was run for up to 40 epochs.

| Dataset   | Metric | Method    | Eta (LR) | Framework  | Score |
| --------- | ------ | --------- | -------  | ---------- | ----- |
| twpos-v03 |    Acc | SGD mom.  |     0.01 | TensorFlow | 89.57 |
| twpos-v03 |    Acc | Adadelta  |       -- | PyTorch    | 89.18 |

