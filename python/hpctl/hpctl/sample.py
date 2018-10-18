from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import map, reduce
from six import with_metaclass

import os
import math
import random
import operator
from copy import deepcopy
from gzip import GzipFile
from random import choice
from functools import partial
from itertools import cycle, product
import numpy as np
from baseline.utils import export as exporter
from baseline.utils import import_user_module, listify, optional_params
from mead.utils import hash_config
from hpctl.utils import Label, register


__all__ = []
export = exporter(__all__)

loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
adjectives = GzipFile(os.path.join(loc, "adjectives.gz")).read().decode('utf-8').rstrip('\n').split('\n')
nouns = GzipFile(os.path.join(loc, "nouns.gz"), "r").read().decode('utf-8').rstrip('\n').split('\n')


## Real Sampling
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
    :param size: int, The number of samples to draw.  :param base: str or int, The base of the log function.
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


## Constrained Sampling
DEFAULT_CONSTRAINTS = {
    'dropout': ['>= 0', '< 1'],
    'mom': ['< 1', '>= 0'],
    'hsz': '% 2 == 0',
    'default': '> 0',
}


def constrained_sampler(sampler, constraints, *args):
    sample = sampler(*args)
    while not all(eval('{} {}'.format(sample, constraint)) for constraint in constraints):
        sample = sampler(*args)
    return sample



# Functions that extract sampling info from a config.
def add_values(example, key, value):
    example[(key,)] = [value['values']]


def add_grid(example, key, value):
    example[(key,)] = value['values']


def add_min_max(example, key, value):
    key = (key,)
    c = value.get('constraints', DEFAULT_CONSTRAINTS.get(key[-1], DEFAULT_CONSTRAINTS['default']))
    example[key] = [listify(c), value['min'], value['max']]


def add_normal(example, key, value):
    key = (key,)
    c = value.get('constraints', DEFAULT_CONSTRAINTS.get(key[-1], DEFAULT_CONSTRAINTS['default']))
    example[key] = [listify(c), value['mu'], value['sigma']]


## Registering Samplers
SAMPLERS = {}


@export
@optional_params
def register_sampler(cls, name=None):
    if cls.name is None:
        cls.name = name if name is not None else cls.__name__
    return register(cls, SAMPLERS, cls.name, "sampler")


# Used to force the _name prop to match between reg and config file.
class NameProp(type):
    _name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value


@export
class Sampler(with_metaclass(NameProp, object)):
    """An object that samples from some distribution.

    :param name: The name of the sampler.
    :param adder: The function that extracts parameters from the config.
    :param sampler: The function that does the real sampling.
    """

    def __init__(self, adder, sampler):
        super(Sampler, self).__init__()
        self._adder = adder
        self.sampler = sampler
        self._values = None

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        return self._values

    @property
    def adder(self):
        return self._adder

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
        vals = ", ".join([".".join(map(str, v)) for v in self._values]) if self._values else "nothing"
        return "<{} sampling on {}>".format(self.name, vals)


@export
@register_sampler
class Grid(Sampler):
    """A special Sampler for going grid search.

    :param adder: The function that extracts parameters from the config.
    """
    _name = 'grid'
    def __init__(self):
        super(Grid, self).__init__(add_grid, None)
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


@export
@register_sampler
class Normal(Sampler):
    _name = 'normal'
    def __init__(self):
        super(Normal, self).__init__(add_normal, partial(constrained_sampler, np.random.normal))


@export
@register_sampler
class Uniform(Sampler):
    _name = "uniform"
    def __init__(self):
        super(Uniform, self).__init__(add_min_max, partial(constrained_sampler, np.random.uniform))

@export
@register_sampler
class Choice(Sampler):
    _name = "choice"
    def __init__(self):
        super(Choice, self).__init__(add_values, random.choice)


@export
@register_sampler
class MinLog(Sampler):
    _name = "min_log"
    def __init__(self):
        super(MinLog, self).__init__(add_min_max, partial(constrained_sampler, min_log_sample))


@export
@register_sampler
class MaxLog(Sampler):
    _name = "max_log"
    def __init__(self):
        super(MaxLog, self).__init__(add_min_max, partial(constrained_sampler, max_log_sample))


@export
@register_sampler
class UniformInt(Sampler):
    _name = "uniform_int"
    def __init__(self):
        super(UniformInt, self).__init__(add_min_max, partial(constrained_sampler, random.randint))


# Whole Config Sampling
@export
class ConfigSampler(object):
    """Sample configs based on a template.

    :param config: dict, The mead config with sampling information.
    :param results: hpctl.results.Results, The dataresults object.
    :param samplers: dict[str] -> hpctl.smapler.Sampler, A mapping of names
        to sampler objects.
    """

    def __init__(self, config, results, samplers):
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
        if not isinstance(config, dict):
            return examples
        for key, value in config.items():
            if isinstance(value, dict):
                if 'hpctl' in value and value['hpctl'] == type_:
                    process_example(examples, key, value)
                else:
                    nested_example = ConfigSampler._find(value, type_, process_example)
                    for k, v in nested_example.items():
                        examples[tuple([key] + list(k))] = v
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        nested_example = ConfigSampler._find(item, type_, process_example)
                        for k, v in nested_example.items():
                            examples[tuple([key, i] + list(k))] = v
        return examples

    @staticmethod
    def _collect(config):
        """Find all sampling types used in the config.

        Is there a way to leverage the _find function to do this?

        :param config: dict, The mead config with the sampling directives.

        :returns: set(str), The names of all used sampling operations.
        """
        examples = set()
        if not isinstance(config, dict):
            return examples
        for key, value in config.items():
            if isinstance(value, dict):
                if 'hpctl' in value:
                    examples.add(value['hpctl'])
                else:
                    nested_example = ConfigSampler._collect(value)
                    for type_ in nested_example:
                        examples.add(type_)
            if isinstance(value, list):
                for i, item in enumerate(value):
                    nested_example = ConfigSampler._collect(item)
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
def get_config_sampler(config, results, samplers=SAMPLERS):
    """Create a ConfigSampler that includes user defined ones.

    ;param config: dict, The mead config with sampling information.
    :param results: hpctl.results.Results, The data results object.
    :param user_samplers: List[str], The names of user defined samplers.
    """
    needed_samplers = ConfigSampler._collect(config)
    samplers = {name: samplers[name]() for name in needed_samplers}
    config_sampler = ConfigSampler(config, results, samplers=samplers)
    return config_sampler
