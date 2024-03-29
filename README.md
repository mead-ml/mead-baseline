# MEAD

MEAD is a library for reproducible deep learning research and fast model
development for NLP. It provides easily extensible abstractions and
implementations for data loading, model development, training, experiment tracking and export to production. 

It also provides implementations of high-performance deep learning models for various NLP tasks, against which newly developed models
can be compared. Deep learning experiments are hard to reproduce, MEAD
provides functionalities to track them. The goal is to allow a researcher to
focus on model development, delegating the repetitive tasks to the library.

[Documentation](https://github.com/dpressel/mead-baseline/blob/master/docs/main.md)

[Tutorials using Colab](https://github.com/dpressel/mead-tutorials)

[MEAD Hub](https://github.com/mead-ml/hub)

## Installation

### Pip

Baseline can be installed as a Python package.

`pip install mead-baseline`

You will need to have
`tensorflow_addons` already installed or have it get installed directly with: 

`pip install mead-baseline[tf2]`

### From the repository

If you have a clone of this repostory and want to install from it:

```
cd layers
pip install -e .
cd ../
pip install -e .
```

This first installs `mead-layers` AKA 8 mile, a tiny layers API containing PyTorch and TensorFlow primitives, locally and then `mead-baseline`

### Dockerhub

We use Github CI/CD to automatically release TensorFlow and PyTorch via this project:

https://github.com/mead-ml/mead-gpu

Links to the latest dockerhub images can be found there

## A Note About Versions

Deep Learning Frameworks are evolving quickly and changes are not always
backwards compatible. We recommend recent versions of whichever framework is being used underneath.  We currently test on TF versions 2.1.0 and 2.4.1.
The PyTorch backend requires at least version 1.3.0, though we recommend using a more recent version.

## Citing

If you use the library, please cite the following paper:

```
@InProceedings{W18-2506,
  author =    "Pressel, Daniel
               and Ray Choudhury, Sagnik
               and Lester, Brian
               and Zhao, Yanjie
               and Barta, Matt",
  title =     "Baseline: A Library for Rapid Modeling, Experimentation and
               Development of Deep Learning Algorithms targeting NLP",
  booktitle = "Proceedings of Workshop for NLP Open Source Software (NLP-OSS)",
  year =      "2018",
  publisher = "Association for Computational Linguistics",
  pages =     "34--40",
  location =  "Melbourne, Australia",
  url =       "http://aclweb.org/anthology/W18-2506"
}
```

MEAD was selected for a Spotlight Poster at the NeurIPS MLOSS workshop in 2018.  [OpenReview link](https://openreview.net/forum?id=r1xEb7J15Q)

### Acknowledgements

- Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

