from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from hpctl.sample import Sampler

def adder(example, key, value):
    example[(key,)] = [value['lam']]

def create_sampler():
    return Sampler("poisson", adder, np.random.poisson)
