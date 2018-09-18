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
def order_json(data):
    """Sort json to a consistent order.

    When you hash json that has the some content but is different orders you get
    different fingerprints.

    In:  hashlib.sha1(json.dumps({'a': 12, 'b':14}).encode('utf-8')).hexdigest()
    Out: '647aa7508f72ece3f8b9df986a206d95fd9a2caf'
    In:  hashlib.sha1(json.dumps({'b': 14, 'a':12}).encode('utf-8')).hexdigest()
    Out: 'a22215982dc0e53617be08de7ba9f1a80d232b23'

    This function sorts json by key so that hashes are consistent.

    Note:
        In our configs we only have lists where the order doesn't matter so we
        can sort them for consistency. This would have to change if we add a
        config field that needs order we will need to refactor this.

    :param data: dict, The json data.

    :returns:
        collections.OrderedDict: The data in a more consistent order.
    """
    new = OrderedDict()
    for key, value in sorted(data.items(), key=lambda x: x[0]):
        if isinstance(value, dict):
            value = order_json(value)
        elif isinstance(value, list):
            value = sorted(value)
        new[key] = value
    return new


KEYS = {
    ('conll_output',),
    ('visdom',),
    ('visdom_name',),
    ('model', 'gpus'),
    ('test_thresh',),
    ('reporting',),
    ('num_valid_to_show',),
    ('train', 'verbose'),
    ('train', 'model_base'),
    ('train', 'model_zip'),
    ('test_batchsz')
}


@export
def remove_monitoring(config, keys=KEYS):
    """Remove config items that don't effect the model.

    When base most things off of the sha1 hash of the model configs but there
    is a problem. Some things in the config file don't effect the model such
    as the name of the `conll_output` file or if you are using `visdom`
    reporting. This strips out these kind of things so that as long as the model
    parameters match the sha1 will too.

    :param config: dict, The json data.
    :param keys: Set[Tuple[str]], The keys to remove.

    :returns:
        dict, The data with certain keys removed.
    """
    c = deepcopy(config)
    for key in keys:
        x = c
        for k in key[:-1]:
            x = x.get(k)
            if x is None:
                break
        else:
            _ = x.pop(key[-1], None)
    return c


@export
def hash_config(config):
    """Hash a json config with sha1.

    :param config: dict, The config to hash.

    :returns:
        str, The sha1 hash.
    """
    stripped_config = remove_monitoring(config)
    sorted_config = order_json(stripped_config)
    json_bytes = json.dumps(sorted_config).encode('utf-8')
    return hashlib.sha1(json_bytes).hexdigest()


@export
def create_logs(label, mead_logs, hpctl_logs):
    """Inject the hpo looging config into the mead logging.

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
