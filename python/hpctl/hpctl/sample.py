from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import map, reduce

import os
import math
import random
import operator
from copy import deepcopy
from gzip import GzipFile
from random import choice
from itertools import cycle, product
import numpy as np
from baseline.utils import export as exporter
from baseline.utils import import_user_module
from mead.utils import hash_config
from hpctl.utils import Label


__all__ = []
export = exporter(__all__)

loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
adjectives = GzipFile(os.path.join(loc, "adjectives.gz")).read().decode('utf-8').rstrip('\n').split('\n')
nouns = GzipFile(os.path.join(loc, "nouns.gz"), "r").read().decode('utf-8').rstrip('\n').split('\n')


@export
def random_name():
    """Generate a human readable random name.

    :return:
        str, The name.
    """
    return "{}_{}_{}".format(choice(adjectives), choice(adjectives), choice(nouns))


def log_sample(min_, max_, size=1, base='e'):
    """Sample from a log scale.

    :param min_: float, The minimum for the sample range.
    :param max_: float, The maximum for the sample range.
    :param size: int, The number of samples to draw.
    :param base: str or int, The base of the log function.
    """
    if base == 'e':
        a = math.log(min_)
        b = math.log(max_)
    else:
        a = math.log(min_, base)
        b = math.log(max_, base)
    r = np.random.uniform(a, b, size)
    if base == 'e':
        return np.exp(r)
    else:
        return np.pow(base, r)


@export
def min_log_sample(min_, max_, size=1, base='e'):
    """Sample from a log scale so many examples are near the min.

    :param min_: float, The minimum for the sample range.
    :param max_: float, The maximum for the sample range.
    :param size: int, The number of samples to draw.
    :param base: str or int, The base of the log function.
    """
    if size == 1:
        return log_sample(min_, max_, size, base).item()
    return log_sample(min_, max_, size, base)


@export
def max_log_sample(min_, max_, size=1, base='e'):
    """Sample from a log scale so many examples are near the max.

    :param min_: float, The minimum for the sample range.
    :param max_: float, The maximum for the sample range.
    :param size: int, The number of samples to draw.
    :param base: str or int, The base of the log function.
    """
    if size == 1:
        return (1 - log_sample(1 - max_, 1 - min_, size, base)).item()
    return 1 - log_sample(1 - max_, 1 - min_, size, base)


@export
class Sampler(object):
    """An object that samples from some distribution.

    :param name: The name of the sampler.
    :param adder: The function that extracts parameters from the config.
    :param sampler: The function that does the real sampling.
    """
    def __init__(self, name, adder, sampler):
        super(Sampler, self).__init__()
        self.name = name
        self.adder = adder
        self.sampler = sampler
        self._values = None

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, val):
        self._values = val

    def sample(self):
        sampled = {}
        for k, args in self.values.items():
            sampled[k] = self.sampler(*args)
        return sampled

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        vals = ", ".join([".".join(v) for v in self._values]) if self._values else "nothing"
        return "<{} sampling on {}.>".format(self.name, vals)


@export
class GridSampler(Sampler):
    """A special Sampler for going grid search.

    :param adder: The function that extracts parameters from the config.
    """
    def __init__(self, adder):
        super(GridSampler, self).__init__("grid", adder, None)
        self.grid_cycles = None

    @Sampler.values.setter
    def values(self, val):
        self._values = val
        self.grid_cycles = cycle(product(*self.values.values()))

    def sample(self):
        sampled = {}
        vals = next(self.grid_cycles)
        for i, k in enumerate(self.values):
            sampled[k] = vals[i]
        return sampled

    def __len__(self):
        """Number of calls needed to use every item in the grid at least once."""
        return reduce(operator.mul, map(len, self.values.values()))


# Functions that extract sampling info from a config.
def add_values(example, key, value):
    example[(key,)] = [value['values']]

def add_grid(example, key, value):
    example[(key,)] = value['values']

def add_min_max(example, key, value):
    example[(key,)] = [value['min'], value['max']]

def add_normal(example, key, value):
    example[(key,)] = [value['mu'], value['sigma']]


DEFAULT_SAMPLERS = {
    'normal': Sampler("normal", add_normal, np.random.normal),
    'grid': GridSampler(add_grid),
    'choice': Sampler("choice", add_values, random.choice),
    'uniform': Sampler("uniform", add_min_max, np.random.uniform),
    'min_log': Sampler("min_log", add_min_max, min_log_sample),
    'max_log': Sampler("max_log", add_min_max, max_log_sample),
}


@export
class ConfigSampler(object):
    """Sample configs based on a template.

    :param config: dict, The mead config with sampling information.
    :param results: hpctl.results.Results, The dataresults object.
    :param samplers: dict[str] -> hpctl.smapler.Sampler, A mapping of names
        to sampler objects.
    """

    def __init__(self, config, results, samplers=DEFAULT_SAMPLERS):
        super(ConfigSampler, self).__init__()
        self.config = config
        self.exp = hash_config(config)
        self.results = results
        self.samplers = samplers
        for sampler in samplers.values():
            sampler.values = ConfigSampler._find(config, sampler.name, sampler.adder)

    @staticmethod
    def _find(config, type_, process_example):
        """Search the config recursively to find all the things to sample.

        :param config: dict, the config to search through.
        :param type_: str, the kind of sample to look for.
        :param process_example: callable, A function to extract params from the sample config

        :returns: dict, keys are tuples (x, y, ...) that are the keys to use to access the element
            in the main config, values are parameters to the sampling function.
        """
        examples = {}
        for key, value in config.items():
            if isinstance(value, dict):
                if 'type' in value and value['type'] == type_:
                    process_example(examples, key, value)
                else:
                    nested_example = ConfigSampler._find(value, type_, process_example)
                    for k, v in nested_example.items():
                        examples[tuple([key] + list(k))] = v
        return examples

    @staticmethod
    def _collect(config):
        examples = set()
        for key, value in config.items():
            if isinstance(value, dict):
                if 'type' in value:
                    examples.add(value['type'])
                else:
                    nested_example = ConfigSampler._collect(value)
                    for type_ in nested_example:
                        examples.add(type_)
        return examples

    def __len__(self):
        """Grid searches are the only ones that have a length."""
        if 'grid' in self.samplers:
            return len(self.samplers['grid'])
        return 0

    def sample(self):
        """Replace values with the sampled ones.

        :returns: tuple, (hpctl.utils.Label dict):
            [0]: The label
            [3]: The config.
        """
        s = deepcopy(self.config)
        for sampler in self.samplers.values():
            sampled = sampler.sample()
            for k, v in sampled.items():
                tmp = s
                for x in k[:-1]:
                    tmp = tmp[x]
                tmp[k[-1]] = v
        hashed = hash_config(s)
        label = Label(self.exp, hashed, random_name())
        return label, s


@export
def build_samplers(names, default_samplers=DEFAULT_SAMPLERS):
    """Create user defined samplers.

    :param names: List[str], The list of user defined sampling classes.

    :returns:
        dict[name] -> hpctl.sampler.Sampler, A mapping of names to user defined
            samplers.
    """
    samplers = {}
    for name in names:
        if name in default_samplers:
            samplers[name] = default_samplers[name]
        else:
            mod = import_user_module("sampler", name)
            samplers[name] = mod.create_sampler()
    return samplers


@export
def get_config_sampler(config, results):
    """Create a ConfigSampler that includes user defined ones.

    ;param config: dict, The mead config with sampling information.
    :param results: hpctl.results.Results, The data results object.
    :param user_samplers: List[str], The names of user defined samplers.
    """
    needed_samplers = ConfigSampler._collect(config)
    samplers = build_samplers(needed_samplers)
    print(samplers)
    config_sampler = ConfigSampler(config, results, samplers=samplers)
    return config_sampler
