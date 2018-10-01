import pytest
from bump import bump_version

def cv(M, m, p, d=None):
    if d is None:
        return '__version__ = "{}.{}.{}"'.format(M, m, p)
    return '__version__ = "{}.{}.{}dev{}"'.format(M, m, p, d)


def test_major_is_updated():
    M = 1
    gold_M = 2
    v = cv(M, 12, 3)
    m, _, _ = bump_version(v, 'major').split(".")
    assert int(m) == gold_M

def test_major_resets_minor():
    gold_m = 0
    m = 15
    v = cv(12, m, 3)
    _, m, _ = bump_version(v, 'major').split(".")
    assert int(m) == gold_m

def test_major_resets_patch():
    gold_p = 0
    p = 15
    v = cv(12, 1, p)
    _, _, p = bump_version(v, 'major').split(".")
    assert int(p) == gold_p

def test_major_resets_dev():
    v = cv(12, 1, 11, '')
    assert 'dev' in v
    res = bump_version(v, 'major').split(".")
    assert 'dev' not in res

    v = cv(12, 1, 15, 12)
    assert 'dev' in v
    res = bump_version(v, 'major')
    assert 'dev' not in res

    _, _, p = res.split('.')
    assert int(p) == 0

def test_minor_is_updated():
    m = 14
    gold_m = 15
    v = cv(12, m, 13)
    _, x, _ = bump_version(v, 'minor').split('.')
    assert int(x) == gold_m

def test_minor_ignores_major():
    M = 12
    gold_M = M
    v = cv(M, 16, 12)
    x, _, _ = bump_version(v, 'minor').split('.')
    assert int(x) == gold_M

def test_minor_resets_patch():
    gold_p = 0
    v = cv(12, 15, 16)
    _, _, x = bump_version(v, 'minor').split('.')
    assert int(x) == gold_p

def test_minor_resets_dev():
    v = cv(12, 1, 11, '')
    assert 'dev' in v
    res = bump_version(v, 'minor').split(".")
    assert 'dev' not in res

    v = cv(12, 1, 15, 12)
    assert 'dev' in v
    res = bump_version(v, 'minor')
    assert 'dev' not in res

    _, _, p = res.split('.')
    assert int(p) == 0

def test_patch_is_updated():
    p = 14
    gold_p = 15
    v = cv(11, 13, p)
    _, _, x = bump_version(v, 'patch').split(".")
    assert int(x) == gold_p

def test_patch_ignores_major():
    gold_M = 1
    v = cv(gold_M, 13, 11)
    x, _, _ = bump_version(v, 'patch').split(".")
    assert int(x) == gold_M

def test_patch_ignores_minor():
    gold_m = 1
    v = cv(13, gold_m, 11)
    _, x, _ = bump_version(v, 'patch').split(".")
    assert int(x) == gold_m

def test_patch_resets_dev():
    p = 1
    gold_p = 2
    v = cv(12, 1, p, '')
    assert 'dev' in v
    res = bump_version(v, 'patch').split(".")
    assert 'dev' not in res

    v = cv(12, 1, p, 12)
    assert 'dev' in v
    res = bump_version(v, 'patch')
    assert 'dev' not in res

    _, _, x = res.split('.')
    assert int(x) == gold_p

def test_dev_is_updated_from_none():
    d = None
    v = cv(1, 1, 1, d)
    res = bump_version(v, 'dev')
    assert res.endswith('dev')

def test_dev_is_updated_from_number():
    d = 12
    gold_d = 13
    v = cv(1, 1, 1, d)
    _, x = bump_version(v, 'dev').split('dev')
    assert int(x) == gold_d

def test_dev_is_updated_from_empty():
    d = ''
    gold_d = 1
    v = cv(1, 1, 1, d)
    _, x = bump_version(v, 'dev').split('dev')
    assert int(x) == gold_d

def test_dev_ignores_major():
    gold_M = 1
    v = cv(gold_M, 13, 11)
    x, _, _ = bump_version(v, 'dev').split(".")
    assert int(x) == gold_M

def test_dev_ignores_minor():
    gold_m = 1
    v = cv(1, gold_m, 13, 11)
    _, x, _ = bump_version(v, 'dev').split(".")
    assert int(x) == gold_m

def test_dev_ignores_patch():
    gold_p = 1
    v = cv(12, 13, gold_p, 13)
    _, _, x = bump_version(v, 'dev').split(".")
    x, _ = x.split('dev')
    assert int(x) == gold_p
