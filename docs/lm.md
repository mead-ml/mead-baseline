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
