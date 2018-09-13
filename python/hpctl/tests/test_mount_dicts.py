import pytest

docker = pytest.importorskip('docker')
from hpctl.dock import create_mount_dict

def test_simple():
    gold = {'a': {'bind': 'a', 'mode': 'ro'}}
    mounts = ['a']
    modes = ['ro']
    mounts = create_mount_dict(mounts, modes)
    assert mounts == gold
