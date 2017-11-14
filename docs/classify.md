# Sentence Classification

There are several models built in to the `baseline` codebase.  These are summarized individually in the sections below

## Convolution Model

This code provides PyTorch, TensorFlow and Keras implementations (as well as an older Lua/Torch7 implementation). 

*Details*

This is inspired by Yoon Kim's paper "Convolutional Neural Networks for Sentence Classification", and before that Collobert's "Sentence Level Approach."  The implementations provided here are basically the Kim static and non-static models.

Temporal convolutional output total number of feature maps is configurable (this also defines the size of the max over time layer, by definition). This code offers several optimization options (adagrad, adadelta, adam and vanilla sgd, with and without momentum).  The Kim paper uses adadelta, which works well, but vanilla SGD and Adam often work well.

Despite the simplicity of this approach, on many datasets this model performs better than other strong baselines such as NBSVM.

Here are some places where this code is known to perform well:

  - Binary classification of sentences (SST2 - SST binary task)
    - Consistently beats RNTN using static embeddings, much simpler model
  - Binary classification of Tweets (SemEval balanced binary splits)
    - Consistent improvement over NBSVM even with char-ngrams included and distance lexicons (compared using [NBSVM-XL](https://github.com/dpressel/nbsvm-xl))
  - Stanford Politeness Corpus
    - Consistent improvement over [extended algorithm](https://github.com/sudhof/politeness) from authors using a decimation split (descending rank heldout)
  - Language Detection (using word and char embeddings)
  - Question Categorization (QA trec)
  - Intent detection

There are some options in each implementation that might vary slightly, but this approach should do at least as well as the original paper.

### With Fine-tuning Embedding (LookupTable) layer

The (default) fine-tuning approach loads the word2vec weight matrix into an Lookup Table.  As we can see from the Kim paper, dynamic/fine-tuning embedding models do not always out-perform static models, However, they tend to do better.

We randomly initialize unattested words and add them to the weight matrix for the Lookup Table.  This can be controlled with the 'unif' parameter in the driver program.

### Static, "frozen" Embedding (LookupTable) layer

The static (no fine-tuning) model usually has decent performance, and the code is very simple to implement from scratch, as long as you have access to a fast convolution operator.  For most cases, fine-tuning is preferred.

### Running It

Early stopping with patience is supported.  There are many hyper-parameters that you can tune, which may yield many different models.  Here is a PyTorch example of parameterization of dynamic embeddings with SGD and the default three filter sizes (3, 4, and 5):

Here is an example running Stanford Sentiment Treebank 2 data with adadelta using pytorch

```
python classify_sentence.py --backend pytorch --clean --optim adadelta --eta 1 --batchsz 50 --epochs 25 \
 --train ../data/stsa.binary.phrases.train \
 --valid ../data/stsa.binary.dev \
 --test ../data/stsa.binary.test \
 --embed /data/xdata/GoogleNews-vectors-negative300.bin \
 --dropout 0.5
```

### Latest Runs

Here are the last observed performance scores using _classify_sentence_ with fine-tuning on the Stanford Sentiment Treebank 2 (SST2)
It was run on the latest code (as of 3/16/2017), with 25 epochs with adadelta as an optimizer:

```

python classify_sentence.py --backend tf --clean --optim adadelta --eta 1 --batchsz 50 --epochs 25 --patience 25 \
 --train ../data/stsa.binary.phrases.train \
 --valid ../data/stsa.binary.dev \
 --test ../data/stsa.binary.test \
 --embed /data/xdata/GoogleNews-vectors-negative300.bin --filtsz 3 4 5 \
 --dropout 0.5

```


Here is an example running the TREC question categorization dataset

```
python classify_sentence.py --optim adadelta --eta 1 --batchsz 10 --epochs 30 --patience 25 \
 --train ../data/trec.nodev.utf8 \
 --valid ../data/trec.dev.utf8 \
 --test ../data/trec.test.utf8 \
 --embed /data/embeddings/GoogleNews-vectors-negative300.bin --filtsz 3 4 5 \
 --dropout 0.5

```

Here is an example running on a preprocessed version of dbpedia with 10% heldout:

```
python classify_sentence.py --optim sgd --eta 0.01 --batchsz 50 --epochs 40 --patience 25 \
 --train /data/xdata/classify/dbpedia_csv/train-tok-nodev.txt \
 --valid /data/xdata/classify/dbpedia_csv/dev-tok.txt \
 --test /data/xdata/classify/dbpedia_csv/test-tok.txt \
 --mxlen 100 \
 --cmotsz 300 \
 --embed /data/embeddings/glove.42B.300d.txt --filtsz 1 2 3 4 5 7 \
 --dropout 0.5
```


| Dataset | TensorFlow | Keras (TF) | PyTorch |
| ------- | ---------- | ---------- | ------- |
| sst2    |       87.9 |      87.4  |  87.9   |
| dbpedia |     99.054 |   --       |  --     | 
| trec-qa |       93.2 |   --       |  92.4   |


Note that these are randomly initialized and these numbers will vary
(IOW, don't assume that one implementation is guaranteed to outperform the others from a single run).

On my laptop, each implementation takes between 29 - 40s per epoch depending on the deep learning framework (TensorFlow and PyTorch are fastest, and about the same speed)

## LSTM Model

Provides a simple LSTM for text classification with PyTorch and TensorFlow

*Details*

The LSTM model provided here expects a time-reversed signal (so that padding will be on the left-side).  The driver program can be passed `--rev 1` to do this time-reversal (you currently must do this when using this model).  The LSTM's final hidden state is then passed to the final layer.  The use of an LSTM instead of parallel convolutional filters is the main differentiator between this model and the default model (CMOT) above.  To request the LSTM classifier instead of the default, pass `--model_type lstm` to the driver program (along with the request for time-reversal).

### Running It

This model is run similarly to the model above.
Here is an example running Stanford Sentiment Treebank 2 data with adadelta using TensorFlow:

```
python classify_sentence.py --clean --optim adadelta --eta 1 --batchsz 50 --epochs 25 --patience 25 \
 --train ../data/stsa.binary.phrases.train \
 --valid ../data/stsa.binary.dev \
 --test ../data/stsa.binary.test \
 --embed /data/xdata/GoogleNews-vectors-negative300.bin \
 --rev 1 \
 --model_type lstm \
 --dropout 0.5
```

### Status

This model is implemented in TensorFlow and PyTorch.  

### Latest Runs

Here are the last observed performance scores using _classify_sentence_ with fine-tuning on various datasets.

| Dataset | TensorFlow | PyTorch | 
| ------- | ---------- | ------- | 
| sst2    |       87.1 |  87.1   |

Note that these are randomly initialized and these numbers will vary

## Neural Bag of Words (NBoW) Model (Max and Average Pooling)

Two different pooling methods for NBoW are supported: max (`--model_type nbowmax`) and average (`--model_type nbow`).  Passing `--layers <N>` defines the number of hidden layers, and passing `--hsz <HU>` defines the number of hidden units for each layer.

### Status

This model is implemented in TensorFlow and PyTorch.  

### Latest Runs

Here are the last observed performance scores using _classify_sentence_ with fine-tuning on the Stanford Sentiment Treebank 2 (sst2) with a single hidden layer (the default), and `hsz=100` (the default)

It was run on the latest code as of 8/24/2017, with 25 epochs with adadelta as an optimizer:

#### NBoW Results

```
python classify_sentence.py --backend pytorch --clean --optim adadelta --eta 1 --batchsz 50 --epochs 25 \
 --train ../data/stsa.binary.phrases.train \
 --valid ../data/stsa.binary.dev \
 --test ../data/stsa.binary.test \
 --embed /data/xdata/GoogleNews-vectors-negative300.bin \
 --dropout 0.5 --model_type nbowmax
```

| model_type | Dataset | TensorFlow | PyTorch | 
|------------| ------- | ---------- | ------- | 
| nbowmax    | sst2    |       82.8 |  84.1   |
| nbow       | sst2    |       84.2 |  82.9   |

Note that these are randomly initialized and these numbers will vary
(IOW, don't assume that one implementation is guaranteed to outperform the others from a single run).
