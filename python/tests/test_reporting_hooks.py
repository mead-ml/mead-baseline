import os
import string
import pytest
from mock import patch
pytest.importorskip('tensorboardX')
import numpy as np
import baseline.reporting
from baseline.reporting import TensorBoardReporting


def random_str(len_=None, min_=5, max_=21):
    if len_ is None:
        len_ = np.random.randint(min_, max_)
    choices = list(string.ascii_letters + string.digits)
    return ''.join([np.random.choice(choices) for _ in range(len_)])


@pytest.fixture
def patches():
    pid = np.random.randint(0, 10)
    with patch('baseline.reporting.os.getpid') as pid_patch:
        pid_patch.return_value = pid
        with patch('tensorboardX.SummaryWriter') as write_patch:
            yield pid_patch, pid, write_patch


def test_no_base_dir(patches):
    p_patch, pid, w_patch = patches
    log_dir = random_str()
    _ = TensorBoardReporting(log_dir=log_dir, flush_secs=2)
    gold = os.path.join('.', log_dir, str(pid))
    w_patch.assert_called_once_with(gold, flush_secs=2)


def test_relative_log_dir(patches):
    p_patch, pid, w_patch = patches
    log_dir = random_str()
    base_dir = random_str()
    _ = TensorBoardReporting(log_dir=log_dir, flush_secs=2, base_dir=base_dir)
    gold = os.path.join(base_dir, log_dir, str(pid))
    w_patch.assert_called_once_with(gold, flush_secs=2)


def test_user_log_dir(patches):
    p_patch, pid, w_patch = patches
    log_dir = random_str()
    log_dir = os.path.join("~", log_dir)
    base_dir = random_str()
    _ = TensorBoardReporting(log_dir=log_dir, flush_secs=2, base_dir=base_dir)
    gold = os.path.join(os.path.expanduser(log_dir), str(pid))
    w_patch.assert_called_once_with(gold, flush_secs=2)


def test_absolute_log_dir(patches):
    p_patch, pid, w_patch = patches
    log_dir = random_str()
    log_dir = os.path.join("/example", log_dir)
    base_dir = random_str()
    _ = TensorBoardReporting(log_dir=log_dir, flush_secs=2, base_dir=base_dir)
    gold = os.path.join(log_dir, str(pid))
    w_patch.assert_called_once_with(gold, flush_secs=2)
