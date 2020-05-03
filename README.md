# Baseline

Baseline is a library for reproducible deep learning research and fast model
development for NLP. The library provides easily extensible abstractions and
implementations for data loading, model development, training and export of deep
learning architectures. It also provides implementations for high-performance,
deep learning models for various NLP tasks, against which newly developed models
can be compared. Deep learning experiments are hard to reproduce, Baseline
provides functionalities to track them. The goal is to allow a researcher to
focus on model development, delegating the repetitive tasks to the library.

[Documentation](https://github.com/dpressel/mead-baseline/blob/master/docs/main.md)

## Installation

### Pip
Baseline can be installed as a Python package.

`pip install mead-baseline`


### From the repository

If you have a clone of this repostory and want to install from it:

```
cd layers
pip install -e .
cd ../
pip install -e .
```

This first installs `mead-layers` (8 mile) locally and then `mead-baseline`

## A Note About Versions

Deep Learning Frameworks are evolving quickly, and changes are not always
backwards compatible. We recommend recent versions of each framework. Baseline
is known to work on most versions of TensorFlow, and is currently being run on
versions between 1.13 and and 2.1 .

The PyTorch backend requires at least version 1.3.0.

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

MEAD/Baseline was selected for a Spotlight Poster at the NeurIPS MLOSS workshop in 2018.  [OpenReview link](https://openreview.net/forum?id=r1xEb7J15Q)
