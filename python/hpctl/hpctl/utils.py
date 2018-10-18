from __future__ import absolute_import, division, print_function, unicode_literals
import six

import os
import json
import hashlib
from collections import Mapping
from functools import partial, update_wrapper
from baseline.utils import export as exporter
from baseline.utils import write_json, register
from mead.utils import convert_path


__all__ = []
export = exporter(__all__)


def wrapped_partial(func, name=None, *args, **kwargs):
    """
    When we use `functools.partial` the `__name__` is not defined which breaks
    our export function so we use update wrapper to give it a `__name__`.
    :param name: A new name that is assigned to `__name__` so that the name
    of the partial can be different than the wrapped function.
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    if name is not None:
        partial_func.__name__ = name
    return partial_func


hpctl_path = export(
    wrapped_partial(
        convert_path,
        loc=os.path.dirname(os.path.realpath(__file__)),
        name='hpctl_path'
    )
)


@export
@six.python_2_unicode_compatible
class Label(Mapping):
    def __init__(self, exp, sha1, name):
        super(Label, self).__init__()
        self.exp = exp
        self.sha1 = sha1
        self.name = name

    @property
    def local(self):
        return "{}/{}".format(self.sha1, self.name)

    def __str__(self):
        return "{}/{}/{}".format(self.exp, self.sha1, self.name)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Label):
            return False
        return (
            self.exp == other.exp and
            self.sha1 == other.sha1 and
            self.name == other.name
        )

    def __hash__(self):
        return hash(str(self))

    def __iter__(self):
        for x in ('exp', 'sha1', 'name'):
            yield x

    def __getitem__(self, item):
        return getattr(self, item)

    def __len__(self):
        return 3

    @classmethod
    def parse(cls, label_str):
        exp, sha1, name = label_str.split("/")
        return cls(exp, sha1, name)


@export
class partialmethod(partial):
    """Python 2.7 doesn't have functools.partialmethod so I made one."""
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return partial(self.func, instance, *(self.args or ()), **(self.keywords or {}))


@export
def create_logs(label, mead_logs, hpctl_logs):
    """Inject the hpctl logging config into the mead logging.

    Note:
        This currently writes it a file that mead need for now.

    :param label: Label, The sha1 and name of the config.
    :param mead_logs: dict, The mead logging config.
    :param hp_logs: dict, The hpctl logging config.

    :returns:
        str, The location of the combined logging file.
    """
    hpctl_logs['label'] = str(label)
    mead_logs['handlers']['hpctl_handler'] = hpctl_logs
    mead_logs['loggers']['baseline.reporting']['handlers'].append('hpctl_handler')
    return mead_logs
