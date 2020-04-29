from six import StringIO

import random
import string
import pytest
import numpy as np
from eight_mile.utils import read_label_first_data, write_label_first_data


def random_str(len_=None, min_=5, max_=21):
    if len_ is None:
        len_ = np.random.randint(min_, max_)
    choices = list(string.ascii_letters + string.digits)
    return "".join([np.random.choice(choices) for _ in range(len_)])


def generate_data():
    labels = [random_str() for _ in range(random.randint(5, 50))]
    texts = [[random_str() for _ in range(random.randint(1, 20))] for _ in range(len(labels))]
    return labels, texts


def test_label_first_data_round_trip():
    data = StringIO()
    labels, texts = generate_data()
    write_label_first_data(data, labels, texts)
    data.seek(0)
    l, ts = read_label_first_data(data)


def test_label_first_data_read_tabs():
    data = StringIO()
    data.write(
        """
1\tdata
2\tdata data
3\tdata\tdata\tdata
    """.lstrip()
    )
    gold_labels = list("123")
    gold_texts = [["data"] * i for i in range(1, 4)]
    data.seek(0)
    l, t = read_label_first_data(data)
    assert l == gold_labels
    assert t == gold_texts


def test_label_first_data_read_space_at_end():
    data = StringIO()
    # Note: The end of the first line in this example has a space after it
    # This is needed for this test
    data.write(
        """
1 data 
2 data data\t
3 data data\tdata\t\t\t
    """.lstrip()
    )
    gold_labels = list("123")
    gold_texts = [["data"] * i for i in range(1, 4)]
    data.seek(0)
    l, t = read_label_first_data(data)
    assert l == gold_labels
    assert t == gold_texts


def test_label_first_data_read_empty_row():
    data = StringIO()
    data.write(
        """
1\tdata

2\tdata data
    """.lstrip()
    )
    gold_labels = list("12")
    gold_texts = [["data"] * i for i in range(1, 3)]
    data.seek(0)
    l, t = read_label_first_data(data)
    assert l == gold_labels
    assert t == gold_texts


def test_label_first_data_read_empty_example():
    data = StringIO()
    data.write(
        """
1\tdata data
2
3 data
    """.lstrip()
    )
    data.seek(0)
    with pytest.raises(ValueError):
        l, t = read_label_first_data(data)


def test_label_first_data_read_single_token():
    data = StringIO()
    data.write(
        """
1 1
2 2
3 3
    """.lstrip()
    )
    data.seek(0)
    l, t = read_label_first_data(data)
    assert l == list("123")
    assert t == [[item] for item in "123"]


def test_write_label_first_data():
    gold = """
1 data
2 data data
3 data data data
5 data data data data data
4 data data data data
    """.strip()
    labels = list("12354")
    texts = [["data"] * int(l) for l in labels]
    data = StringIO()
    write_label_first_data(data, labels, texts)
    data.seek(0)
    assert data.read() == gold
