from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from hpctl.sample import Sampler, register_sampler


def adder(example, key, value):
    example[(key,)] = [value['lam']]


@register_sampler('poisson')
class Poisson(Sampler):
    def __init__(self):
        super(Poisson, self).__init__(adder, np.random.poisson)
