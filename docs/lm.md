# Language Modeling with Recurrent Neural Networks

There are two implemented models (WordLanguageModel, CharCompLanguageModel) based on these two papers:

  - Recurrent Neural Network Regularization (Zaremba, Vinyals, Sutskever) (2014)
    - https://arxiv.org/pdf/1409.2329.pdf
  - Character-Aware Neural Language Models (Kim, Jernite, Sontag, Rush)
    - https://arxiv.org/pdf/1508.06615.pdf

To run the Zaremba model with their "medium regularized LSTM" configuration, early stopping, and pre-trained word vectors:


```
python trainer.py --config config/ptb-med.json
```

## Status

The "medium regularized LSTM" above (_Word Med_ below) has a lower perplexity than the original paper (even the large model).
As noted above, the run above differs in that it uses pre-trained word vectors.

|        Model       | Framework  | Dev    |  Test   |
| ------------------ | ---------- | ------ | ------- |
| Word Med (Zaremba) | TensorFlow | 80.168 | 77.2213 |

_TODO: Add LSTM Char Small Configuration results_

#### Losses and Reporting

The loss that is optimized is the total loss divided by the total number of tokens in the mini-batch (token level loss). This is different than how the loss is calculated in Tensorflow Tutorial but it is how the loss is calculated in awd-lm ([Merity et. al, 2017](https://arxiv.org/abs/1708.02182)), Elmo ([Peters et. al., 2018](https://arxiv.org/abs/1802.05365)), OpenAI GPT ([Radford et. al., 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)), and BERT ([Devlin et. al., 2018](https://arxiv.org/pdf/1810.04805.pdf))

When reporting the loss every nsteps it is the total loss divided by the total number of tokens in the last nstep number of mini-batches. The perplexity is e to this loss.

The epoch loss is the total loss averaged over the total number of tokens in the whole epoch. The perplexity is e to this loss. This results in token level perplexity which is standard reporting in the literature.
