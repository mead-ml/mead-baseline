import pytest

docker = pytest.importorskip('docker')
from hpctl.dock import create_mount_dict, create_mounts, get_container_name


def test_create_mount_dict_single():
    gold = {'a': {'bind': 'a', 'mode': 'ro'}}
    mounts = ['a']
    modes = ['ro']
    mounts = create_mount_dict(mounts, modes)
    assert mounts == gold


def test_create_mount_dict_multiple():
    gold = {'a': {'bind': 'a', 'mode': 'ro'}, 'b': {'bind': 'b', 'mode': 'rw'}}
    mounts = ['a', 'b']
    perms = ['ro', 'rw']
    mounts = create_mount_dict(mounts, perms)
    assert mounts == gold


def test_create_mounts_cwd_included():
    cwd = "cwd"
    gold = {'bind': 'cwd', 'mode': 'rw'}
    mounts = create_mounts([], [], cwd)
    assert cwd in mounts
    assert mounts[cwd] == gold


def test_create_mounts_user_mounts():
    user_mounts = ['a', 'b']
    mounts = create_mounts([], user_mounts, 'cwd')
    for u_mount in user_mounts:
        assert u_mount in mounts
        assert mounts[u_mount]['bind'] == u_mount
        assert mounts[u_mount]['mode'] == 'ro'


def test_create_mounts_default_mounts():
    default_mounts = ['a', 'b']
    mounts = create_mounts(default_mounts, [], 'cwd')
    for d_mount in default_mounts:
        assert d_mount in mounts
        assert mounts[d_mount]['bind'] == d_mount
        assert mounts[d_mount]['mode'] == 'ro'


def test_create_mounts_datacache_mount():
    dcache = '~/.bl-data'
    mounts = create_mounts([], [], 'cwd', datacache=dcache)
    assert dcache in mounts
    assert mounts[dcache]['bind'] == dcache
    assert mounts[dcache]['mode'] == 'rw'


def test_create_mounts_datacache_none():
    dcache = '~/.bl-data'
    mounts = create_mounts([], [], 'cwd', datacache=None)
    assert dcache not in mounts


def test_get_container_name_tf():
    gold = "baseline-tf"
    container = get_container_name("tensorflow")
    assert container == gold


def test_get_container_name_pyt():
    gold = "baseline-pyt"
    container = get_container_name("pytorch")
    assert container == gold


def test_get_container_name_dy():
    gold = "baseline-dy"
    container = get_container_name("dynet")
    assert container == gold
