baseline
=========
Simple but Strong Deep Baseline Algorithms for NLP

# Overview

A few strong, deep baseline algorithms for several common NLP tasks,
including sentence classification and tagging problems.  Considerations are conceptual simplicity, efficiency and accuracy.

After considering other strong, shallow baselines, we have found that even incredibly simple, moderately deep models often perform better.  These models are only slightly more complex to implement than strong baselines such as shingled SVMs and NBSVMs, and support multi-class output easily.

# Sentence Classification using CMOT Model

## Convolution - Max Over Time Architecture (CMOT)

This code provides (at the moment) a pure Lua/Torch7 implementation -- no preprocessing of the dataset with python, nor HDF5 is required!  It depends on a tiny module that can load word2vec in Torch (https://github.com/dpressel/emb) either as a model, or as an nn.LookupTable.  It is important to not that these models can easily be implemented with other deep learning frameworks, and without much work, can also be implemented from scratch!  Over time, we hope to provide alternate framework implementations.

*Details*

This is essentially the Collobert "Sentence Level Approach" architecture, but using off-the-shelf Word2Vec embeddings.  It comes in two flavors, static and dynamic.  This is inspired by Yoon Kim's paper "Convolutional Neural Networks for Sentence Classification", and differs in that it uses a single filter size, doesnt bother with random initialized weights, and doesnt do the multi-channel embeddings.

Hidden unit sizes are configurable.  This code offers several optimization options (adagrad, adadelta, adam and vanilla sgd).  The Kim paper uses adadelta, which seems to work best for fine-tuning, but vanilla SGD often works great for static embeddings.  Input signals are always padded to account for the filter width, so edges are still handled.

Despite the simplicity of these approaches, we have found that on many datasets this performs better than other strong baselines such as NBSVM, ad often performs just as well as the multiple filter approach given by Kim. It seems that the optimization method and the embeddings matter quite a bit: for example, on the Trec QA, we have seen accuracy as high as 94% on static and 94.8% on fine tuning -- much higher than what is reported in the Kim paper, and on par with the SVM (Silva et al. 2011).  This is important since the QA dataset was one of the only sets reported in Kim where shallow methods out-performed deep methods.

Here are some places where CMOT is known to perform well

  - Binary classification of sentences (SST binary task)
  - Binary classification of Tweets (SemEval balanced binary splits)
  - Stanford Politeness Corpus
  - Language Detection
  - Question Categorization (QA trec)
  
## cnn-sentence -- static, no LookupTable layer

This is an efficient implementation of static embeddings, a separate program and routines are provided to preprocess the feature vectors.  Unlike approaches that try to reuse code and then zero gradients on updates, this code preprocesses the training data directly to word vectors.  This means that the first layer of the network is simply TemporalConvolution.  This keeps memory usage on the GPU estremely low, which means it can scale to larger problems.  This model is usually competitive with fine-tuning (it sometimes out-performs fine-tuning), and the code is very simple to implement from scratch (with no deep learning frameworks).

For handling data with high word sparsity, and for data where morphological features are useful, we also provide a very simple solution that occasionally does improve results -- we simply use the average of character vectors generated using word2vec and concatenate this vector.  This is an option in the fixed embeddings version only.  This is useful for problems like Language Detection, for example

For static implementations, batch size and optimization methods can be quite simple.  Often batch sizes of 1-10 with vanilla SGD produce terrific results.

## Dynamic - Fine Tuning Lookup Tables pretrained with Word2Vec

The fine-tuning approach uses the expected LookupTable layer.  It seems that when using fine-tuning, adadelta performs best.  As in the Kim paper suggests, it seems that the Dynamic models do not always out-perform static models, and they have additional baggage due to LookupTable size which may make them cumbersome to use as baselines.

We provide an option to cull non-attested features from the LookupTable for efficiency.

# Structured Prediction using RNNs

This code is useful for tagging tasks, e.g., POS tagging and NER tagging.
Recently, several researchers have proposed using RNNs for tagging, 
particularly LSTMs.  These models do back-propagation through time (BPTT)
and then apply a shared fully connected layer per RNN output to produce a label.
A slight modification is to use Bidirectional LSTMs -- one in the forward direction, and one in the backward direction.

This code is intended to be as simple as possible, and can utilize Justin Johnson's very straightforward, easy to understand [torch-rnn](https://github.com/jcjohnson/torch-rnn) library, or it can use [Element-Research's rnn library](https://github.com/Element-Research/rnn).  When using torch-rnn, we use a convolutional layer to weight share between RNN outputs.  The rnn library makes sequencing easy, so we can simply use a linear layer for that version.

## rnn-tag: Static implementation, input is a temporal feature vector of dense representations

TODO

## rnn-tag-fine: Dynamic (fine-tuning) implementation, input is a sparse vector

TODO
## Results

TODO
