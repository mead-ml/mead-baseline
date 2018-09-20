from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves import intern

import os
import json
import hashlib
from copy import deepcopy
from functools import partial
from collections import OrderedDict, namedtuple
from baseline.utils import export as exporter
from baseline.utils import write_json, wrapped_partial
from mead.utils import convert_path


__all__ = []
export = exporter(__all__)


hpctl_path = export(
    wrapped_partial(
        convert_path,
        loc=os.path.dirname(os.path.realpath(__file__)),
        name='hpctl_path'
    )
)


@six.python_2_unicode_compatible
class Label(object):
    def __init__(self, exp, sha1, human):
        super(Label, self).__init__()
        self.exp = exp
        self.sha1 = sha1
        self.human = human
        intern(self.exp)
        intern(self.sha1)
        intern(self.human)

    @property
    def local(self):
        return "{}@{}".format(self.sha1, self.human)

    def __str__(self):
        return "{}@{}@{}".format(self.exp, self.sha1, self.human)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            self.exp == other.exp and
            self.sha1 == other.sha1 and
            self.human == other.human
        )

    def __hash__(self):
        return hash(str(self))

    @classmethod
    def parse(cls, label_str):
        exp, sha1, human = label_str.split("@")
        return cls(exp, sha1, human)


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
