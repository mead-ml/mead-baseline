import numpy


def assert_dict_almost_equal(expected, actual):
    """
    Assert that dictionaries have almost equal contents, not heavily tested

    """
    try:
        if isinstance(expected, (int, float, numpy.ndarray, list)):
            numpy.testing.assert_allclose(expected, actual)
        elif isinstance(expected, str):
            assert expected == actual
        elif isinstance(expected, dict):
            if set(expected.keys()) != set(actual.keys()):
                assert False
            for key in expected:
                assert_dict_almost_equal(expected[key], actual[key])
        else:
            raise NotImplementedError('only supports dict equality')
    except AssertionError:
        raise AssertionError


if __name__ == "__main__":
    d1 = {'a': 1, 'b': 2, 'c': {'d': 5.3455555678889}, 'e': [2, 4, 5]}
    d2 = {'a': 1, 'b': 2, 'c': {'d': 5.3445555678888}, 'e': [2, 4, 5]}
    assert_dict_almost_equal(d1, d2)
