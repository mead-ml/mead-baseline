# Sentence Classification using CMOT Model

## Convolution - Max Over Time Architecture (CMOT)

This code provides a pure Lua/Torch7 and pure Python PyTorch, TensorFlow and Keras implementations -- no preprocessing of the dataset with python, nor HDF5 is required.  The Lua/Torch code depends on a tiny module that can load word2vec in Torch (https://github.com/dpressel/torchure) either as a model, or as an nn.LookupTable.  It is important to note that these models can easily be implemented with other deep learning frameworks, and without much work, can also be implemented from scratch!  Over time, this package will hopefully provide alternate implementations in other DL Frameworks and programming languages.

*Details*

This is inspired by Yoon Kim's paper "Convolutional Neural Networks for Sentence Classification", and before that Collobert's "Sentence Level Approach."  The implementations provided here are basically the Kim static and non-static models.

This code doesn't implement multi-channel, as this probably does not make sense as a baseline. Temporal convolutional output total number of feature maps is configurable (this is also defines the size of the max over time layer, by definition). This code offers several optimization options (adagrad, adadelta, adam and vanilla sgd).  The Kim paper uses adadelta, which works well, but vanilla SGD often works great for static embeddings.  Input signals are always padded to account for the filter width, so edges are still handled.

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

If you are looking specifically for Yoon Kim's multi-channel model, his code is open source (https://github.com/yoonkim/CNN_sentence) and there is another project on Github from Harvard NLP which recreates it in Torch (https://github.com/harvardnlp/sent-conv-torch).  By focusing on the static and non-static models, we are able to keep this code lean, easy to understand, and applicable as a baseline to a broad range of problems.

There are some options in each implementation that might vary slightly, but this approach should do at least as well as the original paper.

## Fine-tuning Embedding (LookupTable) layer

The (default) fine-tuning approach loads the word2vec weight matrix into an Lookup Table.  As we can see from the Kim paper, dynamic/fine-tuning embedding models do not always out-perform static models, However, they tend to do better.

We randomly initialize unattested words and add them to the weight matrix for the Lookup Table.  This can be controlled with the 'unif' parameter in the driver program.

## Static, "frozen" Embedding (LookupTable) layer

There are several ways to do static embeddings.  One way would be to load a temporal signal comprised of word2vec vectors at each tick.  This can be done by loading the model, and then looking up each word and building a temporal vector out of each lookup.  This will expand the vector in the training data, which will take up more space upfront, and will require transferring more memory to the GPU,
but then bypasses the lookup table altogther.  If you are not fine-tuning, this means you could pre-compute your feature vectors all the way to post-embedding layer. 
This would mean that the first layer of the network would simply be a 1D Convolution.  I used to have separate programs for demonstrating this directly, but for the purposes of demonstration,
this isnt necessary, and created a lot more redundant code.  Instead, I eventually made all programs support a 'static' command-line option that "freezes" the embedding (LUT) layer,
not allowing the error to back-propagate and update the weights.
When this is exercised currently, the 'unif' parameter is ignored, forcing unattested vectors to zeros.

The static (no fine-tuning) model usually has decent performance, and the code is very simple to implement from scratch, as long as you have access to a fast convolution operator.  For most cases, fine-tuning is preferred.

## Running It

Early stopping with patience is supported.  There are many hyper-parameters that you can tune, which may yield many different models.  Here is a PyTorch example of parameterization of dynamic embeddings with SGD and the default three filter sizes (3, 4, and 5):

Here is an example running Stanford Sentiment Treebank 2 data with adadelta

```
python classify_sentence.py --backend pytorch --clean --optim adadelta --eta 0.01 --batchsz 50 --epochs 25 --patience 25 --train ./data/stsa.binary.phrases.train --valid ./data/stsa.binary.dev --test ./data/stsa.binary.test --embed /data/xdata/GoogleNews-vectors-negative300.bin --filtsz 3 4 5 --dropout 0.5
```

(Note that these are already the default arguments!)

## Status

This model is implemented in TensorFlow, Keras, Torch, and PyTorch.  Currently, the PyTorch model does not support 'static' embeddings.  The Keras model currently does not use an 'eta' parameter.  Weight initialization techniques vary slightly across implementations at the moment.

All of the models should typically achieve the dynamic fine-tune results on SST from the Kim paper, though there is some slight variation between runs (I have seen accuracy as high as *88.36%!*).

### Latest Runs

Here are the last observed performance scores using _classify_sentence_ with fine-tuning on the Stanford Sentiment Treebank 2 (SST2)
It was run on the latest code as of 3/16/2017, with 25 epochs, a learning rate of 0.01 and adadelta as an optimizer:

| Dataset | TensorFlow | Keras (TF) | PyTorch | Torch7 |
| ------- | ---------- | ---------- | ------- | ------ |
| SST2    |      87.70 |      87.4  |  87.5   | 87.095 |

Note that these are randomly initialized and these numbers will vary
(IOW, don't assume that one implementation is guaranteed to outperform the others from a single run).

On my laptop, each implementation takes between 29 - 40s per epoch.
