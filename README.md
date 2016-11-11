baseline
=========
Simple, Strong Deep-Learning Baselines for NLP in several frameworks

Stand-alone baselines implemented with multiple deep learning tools, including CNN sentence modeling, RNN/LSTM-based tagging, seq2seq, and siamese networks for similarity analysis.

# Overview

A few strong, deep baseline algorithms for several common NLP tasks,
including sentence classification and tagging problems.  Considerations are conceptual simplicity, efficiency and accuracy.  This is intended as a simple to understand reference for building strong baselines with deep learning.

After considering other strong, shallow baselines, we have found that even incredibly simple, moderately deep models often perform better.  These models are only slightly more complex to implement than strong baselines such as shingled SVMs and NBSVMs, and support multi-class output easily.  Additionally, they are (hopefully) the first "deep learning" thing you might think of for a certain type of problem.  Using these stronger baselines as a reference point hopefully yields more productive algorithms and experimentation.

Each algorithm is in a separate sub-directory, and is fully contained, even though this means that there is some overlap in the routines.  This is done for ease of use and experimentation.

# Sentence Classification using CMOT Model

## Convolution - Max Over Time Architecture (CMOT)

This code provides a pure Lua/Torch7 and pure Python Tensorflow and Keras implementations -- no preprocessing of the dataset with python, nor HDF5 is required.  The Torch code depends on a tiny module that can load word2vec in Torch (https://github.com/dpressel/torchure) either as a model, or as an nn.LookupTable.  It is important to note that these models can easily be implemented with other deep learning frameworks, and without much work, can also be implemented from scratch!  Over time, this package will hopefully provide alternate implementations in other DL Frameworks and programming languages.

When the GPU is used, the code *assumes that cudnn (>= R4) is available* and installed. This is because the performance gains over the 'cunn' implementation are significant (e.g., 3 minutes -> 30 seconds).

*Details*

This is inspired by Yoon Kim's paper "Convolutional Neural Networks for Sentence Classification", and before that Collobert's "Sentence Level Approach."  The implementations provided here are basically the Kim static and non-static models.

To explicitly specify filter widths at the same time, pass in -filtsz {3,4,5} at the command line for the Torch driver, or "3,4,5" for the Python drivers (these values are the default for the driver programs, so you do not have to pass them if these are the filter sizes you desire.

This code doesn't implement multi-channel, as this probably does not make sense as a baseline. It does support adding a hidden projection layer (if you pass hsz), which is kind of like the "Sentence Level Approach" in the Collobert et al. paper, "Natural Language Processing (Almost) from Scratch"

Temporal convolutional output total number of feature maps is configurable (this is also defines the size of the max over time layer, by definition).  This code offers several optimization options (adagrad, adadelta, adam and vanilla sgd).  The Kim paper uses adadelta, which works well, but vanilla SGD often works great for static embeddings.  Input signals are always padded to account for the filter width, so edges are still handled.

Despite the simplicity of these approaches, we have found that on many datasets this performs better than other strong baselines such as NBSVM.

Here are some places where this code is known to perform well:

  - Binary classification of sentences (SST2 - SST binary task)
    - Consistently beats RNTN using static embeddings, much simpler model
  - Binary classification of Tweets (SemEval balanced binary splits)
    - Consistent improvement over NBSVM even with char-ngrams included and distance lexicons (compared using [NBSVM-XL](https://github.com/dpressel/nbsvm-xl))
  - Stanford Politeness Corpus
    - Consistent improvement over [extended algorithm](https://github.com/sudhof/politeness) from authors using a decimation split (descending rank heldout)
  - Language Detection (using word and char embeddings)
  - Question Categorization (QA trec)

This architecture doesn't seem to perform especially well on long posts compared to NBSVM or even SVM.  However, this pattern is used to good effect as a compositional portion of larger models by various researchers.

If you are looking specifically for Yoon Kim's multi-channel model, his code is open source (https://github.com/yoonkim/CNN_sentence) and there is another project on Github from Harvard NLP which recreates it in Torch (https://github.com/harvardnlp/sent-conv-torch).  By focusing on the static and non-static models, we are able to keep this code lean, easy to understand, and applicable as a baseline to a broad range of problems.

There are some options in each implementation that might vary slightly, but each approach implementation has been tested separately.

## Fine-tuning Embedding (LookupTable) layer
The (default) fine-tuning approach loads the word2vec weight matrix into an Lookup Table.  As we can see from the Kim paper, dynamic/fine-tuning embedding models do not always out-perform static models, However, they tend to do better.

We randomly initialize unattested words and add them to the weight matrix for the Lookup Table.  This can be controlled with the 'unif' parameter in the driver program.

## Static, "frozen" Embedding (LookupTable) layer

There are several ways to do static embeddings.  One way would be to load a temporal signal comprised of word2vec vectors at each tick.  This can be done by loading the model, and then looking up each word and building a temporal vector of each lookup.  This will expand the vector in the training data, which will take up more space upfront, but then bypasses the lookup table altogther.  If you are not fine-tuning, this means you could pre-compute your feature vectors all the way to post-embedding layer.  This would mean that the first layer of the network would simply be a 1D Convolution.  This could keep memory usage on the GPU estremely low, which means it could potentially scale to larger problems.  I used to have separate programs for demonstrating this directly, but for the purposes of demonstration, this isnt probably necessary, and created a lot more redundant code.  Instead, I eventually made all programs support a 'static' command-line option that "freezes" the embedding (LUT) layer, not allowing the error to back-propagate and update the weights. When this is exercised currently, the 'unif' parameter is ignored, forcing unattested vectors to zeros.

The static (no fine-tuning) model usually has decent performance, and the code is very simple to implement from scratch, as long as you have access to a fast convolution operator.

For handling data with high word sparsity, and for data where morphological features are useful, we also provide a very simple solution that occasionally does improve results -- we simply use the average of character vectors passed in from a word2vec-style binary file and concatenate this word representation to the word2vec vector.  This is an option in the fixed embeddings version only.  This is useful for problems like Language Detection in Twitter, for example.

## Running It

Early stopping with patience is supported.  There are many hyper-parameters that you can tune, which may yield many different models.  Here is a Torch example of parameterization of static embeddings with SGD and the default three filter sizes (3, 4, and 5):

```
th classify_sentence.lua -static -eta 0.01 -batchsz 10 -decay 1e-9 -epochs 20 -train ../data/TREC.train.all -eval ../data/TREC.test.all -embed /data/xdata/GoogleNews-vectors-negative300.bin
```

And with Tensorflow or Keras
```
python classify_sentence.py  --static --eta 0.01 --batchsz 10 -epochs 20 --train ../data/TREC.train.all --test ../data/TREC.test.all --embed /data/xdata/GoogleNews-vectors-negative300.bin
```

Here is an example of parameterization of dynamic fine tuning (classify_sentence.lua) with SGD to train TREC QA set

```
th classify_sentence.lua -optim adadelta -patience 20 -batchsz 50 -epochs 25 -train ../data/TREC.train.all -eval ../data/TREC.test.all -embed /data/xdata/GoogleNews-vectors-negative300.bin
```

And in Tensorflow and Keras versions, its basically the same

```
python classify_sentence.py --optim adadelta --batchsz 50 --epochs 25 --patience 25 --train ../data/TREC.train.all --test ../data/TREC.test.all -embed /data/xdata/GoogleNews-vectors-negative300.bin
```

Here is an example running Stanford Sentiment Treebank 2 data with adadelta

```

th classify_sentence.lua -clean -optim adadelta -batchsz 50 -epochs 25 -patience 25 -train ./data/stsa.binary.phrases.train -valid ./data/stsa.binary.dev -eval ./data/stsa.binary.test -embed /data/xdata/GoogleNews-vectors-negative300.bin -filtsz "{3,4,5}"
```
In Tensorflow or Keras:

```
python2.7 classify_sentence.py --clean --optim adadelta --eta 0.004 --batchsz 50 --epochs 25 --patience 25 --train ./data/stsa.binary.phrases.train --valid ./data/stsa.binary.dev --test ./data/stsa.binary.test --embed /data/xdata/GoogleNews-vectors-negative300.bin --filtsz "3,4,5" --dropout 0.5
```

All of the models should typically achieve the dynamic fine-tune results on SST from the Kim paper, though there is variation between runs.  I have found that random uniform initialization of the convolutional layers with Glorot initializatoin on the fully-connected layers tends to work well, so that is what happens here in Tensorflow (and is default in Keras).

## Restoring the Model

In Torch and in Keras, restoring the model is trivial, but with TensorFlow there is a little more work.  The CNN classes are set up to handle this save and restore, which includes reloading the graph, and then reinitializing the model, along with labels and feature index.

[Here is a gist](https://gist.github.com/dpressel/767db3d1b5b6bf263ebfd7fed716be96) for performing classification on a previously trained classify_sentence model, and [here is one](https://gist.github.com/dpressel/61ccf0e89c5a5c3e801d7eb15246b78f) for restoring the static model.

# Structured Prediction using RNNs

This code is useful for tagging tasks, e.g., POS tagging, chunking and NER tagging.  Recently, several researchers have proposed using RNNs for tagging, particularly LSTMs.  These models do back-propagation through time (BPTT)
and then apply a shared fully connected layer per RNN output to produce a label.
A common modification is to use Bidirectional LSTMs -- one in the forward direction, and one in the backward direction.  This usually improves the resolution of the tagger significantly.

This code is intended to be as simple as possible, and can utilize Justin Johnson's very straightforward, easy to understand [torch-rnn](https://github.com/jcjohnson/torch-rnn) library, or it can use [Element-Research's rnn library](https://github.com/Element-Research/rnn).  When using torch-rnn, we use a convolutional layer to weight share between RNN outputs.  The rnn library makes sequencing easy, so we can simply use a linear layer for that version.  This approach does not currently use a Sentence Level Likelihood as described in Collobert's various works using convolutional taggers.

## rnn-tag: Static implementation, input is a temporal feature vector of dense representations

Twitter is challenging data source for tagging problems.  The [TweetNLP project](http://www.cs.cmu.edu/~ark/TweetNLP) includes hand annotated POS data. The original taggers used for this task are described [here](http://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf).  The baseline that they compared their algorithm against got 83.38% accuracy.  The final model got 89.37% accuracy with many custom features.  Below, our simple BLSTM baseline with no custom features, no fine tuning of embeddings, and a very coarse approach to compositional character to word modeling still gets *89%* accuracy.  Without any character vectors, the model still gets 88.27% accuracy.

This has been tested on oct27 train+dev and test splits (http://www.cs.cmu.edu/~ark/TweetNLP), using custom word2vec embedddings generated from ~32M tweets including s140 and the oct27 train+dev data.  Some of the data was sampled and preprocessed to have placeholder words for hashtags, mentions and URLs to be used as backoffs for words of those classes which are not found.  It also employs character vectors taken from splitting oct27 train+dev and s140 data and uses them to build averaged word vectors over characters.  This is a simple way of accounting for morphology and sparse terms while being simple enough to be a strong baseline.

```
th rnn-tag.lua -rnn blstm -eta .3 -optim adagrad -epochs 60 -embed /data/xdata/oct-s140clean-uber.cbow-bin -cembed /data/xdata/oct27-s140-char2vec-cbow-50.bin -hsz 100 -train /data/xdata/twpos-data-v0.3/oct27.splits/oct27.traindev -eval /data/xdata/twpos-data-v0.3/oct27.splits/oct27.test
```

## rnn-tag-fine: Dynamic (fine-tuning) implementation, input is a sparse vector

Right now, the fine tuning version only supports word based tagging -- no character level backoff yet.  This will hopefully be fixed in the near future.

# Seq2Seq

Encoder-decoder frameworks have been used for Statistical Machine Translation, Image Captioning, ASR, Conversation Modeling, Formula Generation and many other applications.  Seq2seq is a type of encoder-decoder implementation, where the encoder is some sort of RNN, and the memories (sometimes known as the "thought vector") are transferred over to the decoder, also an RNN, essentially making this a conditional language model.  The code as it is written here is with text in mind, and supports multiple types of RNNs, including GRUs and LSTMs, as well as stacked layers.  The TensorFlow version also supports attention.

## seq2seq: Phrase in, phrase-out, 2 lookup table implementation, input is a temporal vector, and so is output

This code implements seq2seq with mini-batching (as in other examples) using adagrad, adadelta, sgd or adam.  It supports two vocabularies, and takes word2vec pre-trained models as input, filling in words that are attested in the dataset but not found in the pre-trained models.  It uses dropout for regularization.

For any reasonable size data, this really needs to run on the GPU for realistic training times.

# Distance metrics using Siamese Networks

Siamese networks have been shown to be useful for tasks such as paraphrase detection, and are generally helpful for learning similarity/distance metrics.  The siamese network provided here is a convolutional neural net, based on the classify_sentence model above.  It uses an L2 (pairwise distance) metric function and a contrastive loss function to determine a distance between pairs.  For example, for a paraphrase corpus, the data will include 2 sentences and a label (0,1) stating whether or not the two documents are paraphases.  The Siamese network then learns a distance mapping from this data.

# siamese-fine: Parallel CNNs with shared weights, using Word2Vec input + fine-tuning with an L2 loss function

This example shows how to use a simple flat CNN with fine-tuning and ReLU activation (with dropout):

```
th siamese-fine.lua -optim adagrad -epochs 50 -patience 25 -cmotsz 100 -embed /data/xdata/oct-s140clean-uber.cbow-bin -train /data/xdata/para/tw/train.txt -valid /data/xdata/para/tw/dev.txt -eval /data/xdata/para/tw/test.txt
```
Same, but with an additional linear hidden layer

```
th siamese-fine.lua -optim adagrad -epochs 50 -patience 25 -cmotsz 100 -hsz 80 -embed /data/xdata/oct-s140clean-uber.cbow-bin -train /data/xdata/para/tw/train.txt -valid /data/xdata/para/tw/dev.txt -eval /data/xdata/para/tw/test.txt

```
