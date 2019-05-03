import pytest
import json
import os
from xpctl.backend.mongo import MongoRepo
import pickle


@pytest.fixture
def setup():
    creds = json.load(open(os.path.expanduser(json.load(open('xpctlcred-loc.json'))['loc'])))
    return creds


def test_dto_mongo(setup):
    creds = setup
    input_ = json.load(open('data/mongosample3.json'))
    m = MongoRepo(**creds)
    rs = m.mongo_to_experiment_set(input_, event_type=None, metrics=[])
    rs_stored = pickle.load(open('data/mongosample3.resultset.pickle', 'rb'))
    assert len(rs) == len(rs_stored)
    matched = 0
    for d1 in rs:
        for d2 in rs_stored:
            if d1.__dict__ == d2.__dict__:
                matched += 1
    assert len(rs) == matched


def test_grouping():
    rs = pickle.load(open('data/mongosample3.resultset.pickle', 'rb'))
    lengths = {
        '_id': {
            '5af36f9bb5536c60d1e2ccc1': 1,
            '5af3c937b5536c75fb1529bc': 1,
            '5b074775b5536c4bc124d95f': 1,
            '5b0eb313b5536c45af5f9d86': 1,
            '5b1aacb933ed5901dc545af8': 1},
        'sha1': {
            '1a870728e04470a07643c5d9cff33329c004751f': 1,
            '67105e2108885c5ee08e211537fbda602f2ba254': 1,
            '8ab6ab6ee8fdf14b111223e2edf48750c30c7e51': 3},
        'username': {
            'x': 2,
            'y': 1,
            'z': 2
        }
    }

    d = {}
    for prop in ['username', '_id', 'sha1']:
        d[prop] = {k: v.__dict__['length'] for k, v in rs.groupby(prop)}
    assert d == lengths


