baseline
=========
Simple but Strong Deep Baseline Algorithms for NLP

# Overview

A few strong, deep baseline algorithms for several common NLP tasks,
including sentence classification and tagging problems.  Considerations are conceptual simplicity, efficiency and accuracy.

After considering other strong, shallow baselines, we have found that even incredibly simple, moderately deep models often perform better.  These models are only slightly more complex to implement than strong baselines such as shingled SVMs and NBSVMs, and support multi-class output easily.  Additionally, they are (hopefully) the first thing you might think of for a certain type of problem.  Using these stronger baselines as a reference point hopefully yields more productive algorithms and experimentation.

# Sentence Classification using CMOT Model

## Convolution - Max Over Time Architecture (CMOT)

This code provides (at the moment) a pure Lua/Torch7 implementation -- no preprocessing of the dataset with python, nor HDF5 is required!  It depends on a tiny module that can load word2vec in Torch (https://github.com/dpressel/emb) either as a model, or as an nn.LookupTable.  It is important to note that these models can easily be implemented with other deep learning frameworks, and without much work, can also be implemented from scratch!  Over time, this package will hopefully provide alternate implementations in other DL Frameworks and programming languages.

*Details*

This is essentially the Collobert "Sentence Level Approach" architecture, but using off-the-shelf Word2Vec embeddings.  It comes in two flavors, static and dynamic.  This is inspired by Yoon Kim's paper "Convolutional Neural Networks for Sentence Classification", and differs in that it uses a single filter size, doesn't bother with random initialized embeddings options, and doesn't do the multi-channel embeddings.

Hidden unit sizes are configurable.  This code offers several optimization options (adagrad, adadelta, adam and vanilla sgd).  The Kim paper uses adadelta, which seems to work best for fine-tuning, but vanilla SGD often works great for static embeddings.  Input signals are always padded to account for the filter width, so edges are still handled.

Despite the simplicity of these approaches, we have found that on many datasets this performs better than other strong baselines such as NBSVM, and often performs just as well as the multiple filter approach given by Kim. It seems that the optimization method and the embeddings matter quite a bit. For example, on the Trec QA, we tend to see around the same performance for fine-tuning as the Kim paper (93.6%), but also get the same using SGD with no fine tuning.  Due to random shuffling, we have seen accuracy as high as 94% on static and 94.8% on fine tuning -- much higher than what is reported in the Kim paper.

Here are some places where CMOT is known to perform well

  - Binary classification of sentences (SST2 - SST binary task)
    - Consistently beats RNTN using static embeddings, much simpler model
  - Binary classification of Tweets (SemEval balanced binary splits)
    - Consistent improvement over NBSVM even with char-ngrams included and distance lexicons (compared using [NBSVM-XL](https://github.com/dpressel/nbsvm-xl))
  - Stanford Politeness Corpus
    - Consistent improvement over [extended algorithm](https://github.com/sudhof/politeness) from authors using a fair split (descending rank heldout)
  - Language Detection (using word and char embeddings)
  - Question Categorization (QA trec) (93.6-94% static using SGD, 93.6-94.8% dynamic using adadelta)
  
## cnn-sentence -- static, no LookupTable layer

This is an efficient implementation of static embeddings, a separate program and routines are provided to preprocess the feature vectors.  Unlike approaches that try to reuse code and then zero gradients on updates, this code preprocesses the training data directly to word vectors.  This means that the first layer of the network is simply TemporalConvolution.  This keeps memory usage on the GPU estremely low, which means it can scale to larger problems.  This model is usually competitive with fine-tuning (it sometimes out-performs fine-tuning), and the code is very simple to implement from scratch (with no deep learning frameworks).

For handling data with high word sparsity, and for data where morphological features are useful, we also provide a very simple solution that occasionally does improve results -- we simply use the average of character vectors generated using word2vec and concatenate this vector.  This is an option in the fixed embeddings version only.  This is useful for problems like Language Detection, for example

The goal of this code is to present a lean but very strong baseline while remaining simple and efficient. If you were looking for Yoon Kim's approach, his code is open source (https://github.com/yoonkim/CNN_sentence) and there is another project on Github from Harvard NLP which recreates it in Torch (https://github.com/harvardnlp/sent-conv-torch).  As described above, the Kim model actually includes multiple approaches, which might sometimes out-perform this simple baseline. Note that if you are going to use static embeddings though, code that treats all cases the same cannot be as efficient as what is implemented here since it would use the same code as for dynamic models, which requires that torch backprops through the weights, after which they would need to zero out the gradients.  This exact approach can be seen in the Harvard code.

Also note that for static implementations, batch size and optimization methods can be quite simple.  Often batch sizes of 1-10 with vanilla SGD produce terrific results.

## Dynamic - Fine Tuning Lookup Tables pretrained with Word2Vec

The fine-tuning approach uses the expected LookupTable layer.  It seems that when using fine-tuning, adadelta performs best.  As we can see from the Kim paper, it seems that the dynamic fine-tuning models do not always out-perform static models, and they have additional baggage due to LookupTable size which may make them cumbersome to use as baselines.  However, if tuned properly, they often can out-perform the static models slightly.

We provide an option to cull non-attested features from the LookupTable for efficiency.

## Running It

Early stopping with a single patience is used.  There are many hyper-parameters that you can tune, which may yield many different models.  Due to random shuffling performed during training, runs may achieve slightly different performance each run.  Therefore multiple runs are suggested for each configuration.

You can get some sample data from the Harvard NLP [sent-conv-torch project](https://github.com/harvardnlp/sent-conv-torch).

Here is an example of parameterization of static embeddings (cnn-sentence.lua) with SGD, achieving final accuracy of *93.6-94%*

```
th cnn-sentence.lua -eta 0.01 -batchsz 10 -decay 1e-9 -epochs 200 -train /home/dpressel/dev/work/sent-conv-torch/data/TREC.train.all -eval /home/dpressel/dev/work/sent-conv-torch/data/TREC.test.all -embed /data/xdata/GoogleNews-vectors-negative300.bin
```

Here is an example of parameterization of dynamic fine tuning (cnn-sentence-fine.lua) with SGD achieving final accuracy of *93.6-94.8%*

```
th cnn-sentence-fine.lua -cullunused -optim adadelta -patience 50 -batchsz 10 -decay 1e-9 -epochs 1000 -train /home/dpressel/dev/work/sent-conv-torch/data/TREC.train.all -eval /home/dpressel/dev/work/sent-conv-torch/data/TREC.test.all -embed /data/xdata/GoogleNews-vectors-negative300.bin
```

# Structured Prediction using RNNs

This code is useful for tagging tasks, e.g., POS tagging, chunking and NER tagging.  Recently, several researchers have proposed using RNNs for tagging, 
particularly LSTMs.  These models do back-propagation through time (BPTT)
and then apply a shared fully connected layer per RNN output to produce a label.
A common modification is to use Bidirectional LSTMs -- one in the forward direction, and one in the backward direction.  This usually improves the resolution of the tagger significantly.

This code is intended to be as simple as possible, and can utilize Justin Johnson's very straightforward, easy to understand [torch-rnn](https://github.com/jcjohnson/torch-rnn) library, or it can use [Element-Research's rnn library](https://github.com/Element-Research/rnn).  When using torch-rnn, we use a convolutional layer to weight share between RNN outputs.  The rnn library makes sequencing easy, so we can simply use a linear layer for that version.  This approach does not currently use a Sentence Level Likelihood as described in Collobert's various works using convolutional taggers.

## rnn-tag: Static implementation, input is a temporal feature vector of dense representations

Twitter is challenging to build a tagger for.  The [TweetNLP project](http://www.cs.cmu.edu/~ark/TweetNLP) includes hand annotated POS data. The original taggers used for this task are described [here](http://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf).  The baseline that they compared their algorithm against got 83.38% accuracy.  The Stanford Tagger (at the time of that paper's publication) got 85.85% accuracy.  The final model got 89.37% accuracy with many custom features.  Below, our simple BLSTM baseline with no custom features, and no fine tuning of embeddings gets *88.67%* accuracy.  Without any character vectors, the model still gets 88.27% accuracy.

This has been tested on oct27 train+dev and test splits (http://www.cs.cmu.edu/~ark/TweetNLP), using custom word2vec embedddings generated from ~32M tweets including s140 and the oct27 train+dev data.  Some of the data was sampled and preprocessed to have placeholder words for hashtags, mentions and URLs to be used as backoffs for words of those classes which are not found.  It also employs character vectors taken from splitting oct27 train+dev and s140 data and uses them to build averaged word vectors over characters.  This is a simple way of accounting for morphology and sparse terms while being simple enough to be a strong baseline.


```
th rnn-tag.lua -usernnpkg -rnn blstm -eta .32 -optim adagrad -epochs 60 -embed /data/xdata/oct-s140clean-uber.cbow-bin -cembed /data/xdata/oct27-s140-char2vec-cbow-50.bin -hsz 100 -train /data/xdata/twpos-data-v0.3/oct27.splits/oct27.traindev -eval /data/xdata/twpos-data-v0.3/oct27.splits/oct27.test
```

## rnn-tag-fine: Dynamic (fine-tuning) implementation, input is a sparse vector

Right now, the fine tuning version only supports word based tagging -- no character level backoff yet.  This will hopefully be fixed in the near future.