import string
import random
from collections import Counter
import pytest
from mock import patch, call
import numpy as np
from eight_mile.bleu import (
    n_grams,
    count_n_grams,
    find_closest,
    corpora_lengths,
    max_gold_n_gram_counts,
    count_matches,
    count_possible,
    geometric_mean,
    brevity_penalty,
    _read_references,
    _read_lines,
)


def random_str(len_=None, min_=5, max_=21):
    if len_ is None:
        len_ = np.random.randint(min_, max_)
    choices = list(string.ascii_letters + string.digits)
    return "".join([np.random.choice(choices) for _ in range(len_)])


def test_find_closest_above():
    pred_len = np.random.randint(10, 20)
    input_lens = pred_len + np.random.randint(5, 10, size=np.random.randint(2, 4))
    gold = np.min(input_lens)
    input_ = [[""] * input_len for input_len in input_lens]
    res = find_closest(pred_len, input_)
    assert res == gold


def test_find_closest_below():
    pred_len = np.random.randint(10, 20)
    input_lens = pred_len - np.random.randint(5, 10, size=np.random.randint(2, 4))
    gold = np.max(input_lens)
    input_ = [[""] * input_len for input_len in input_lens]
    res = find_closest(pred_len, input_)
    assert res == gold


def test_find_closest_tie():
    pred_len = np.random.randint(10, 20)
    offset = np.random.randint(5, 10)
    above = pred_len + offset
    below = pred_len - offset
    input_ = [[""] * input_len for input_len in (above, below)]
    gold = below
    res = find_closest(pred_len, input_)
    assert res == gold


def test_corpora_lengths():
    pred_lens = np.random.randint(2, 20, size=np.random.randint(100, 200))
    gold_pred = np.sum(pred_lens)
    gold_lens = np.random.randint(2, 20, size=len(pred_lens))
    gold_gold = np.sum(gold_lens)
    preds = [[""] * p for p in pred_lens]
    golds = [[[""] * g] for g in gold_lens]
    with patch("eight_mile.bleu.find_closest") as find_patch:
        find_patch.side_effect = gold_lens
        pred_guess, gold_guess = corpora_lengths(preds, golds)
    assert pred_guess == gold_pred
    assert gold_guess == gold_gold


def test_max_gold_counts():
    # Create a gold that has strings and a max count for each
    gold = Counter()
    for _ in range(np.random.randint(5, 10)):
        gold[random_str()] = np.random.randint(10, 20)
    # For each word create a counter that will have the max value for it
    # With some probability (0.5) add other words to the counter with
    # A smaller value (than gold for that string) for the count
    counters = []
    for word, count in gold.items():
        counter = Counter()
        counter[word] = count
        for word2, count2 in gold.items():
            if word == word2:
                continue
            if np.random.rand() > 0.5:
                counter[word2] = count2 - np.random.randint(1, count2)
        counters.append(counter)
    random.shuffle(counters)
    # Have the reduce work on all of these counters.
    with patch("eight_mile.bleu.count_n_grams") as count_mock:
        count_mock.side_effect = counters
        res = max_gold_n_gram_counts([""] * len(gold), None)
    assert res == gold


def test_max_gold_counts_calls():
    input_ = [chr(i + 97) * l for i, l in enumerate(np.random.randint(10, 20, size=np.random.randint(5, 10)))]
    n = np.random.randint(2, 6)
    golds = [call(i, n) for i in input_]
    with patch("eight_mile.bleu.count_n_grams") as count_mock:
        _ = max_gold_n_gram_counts(input_, n)
    assert count_mock.call_args_list == golds


def test_n_grams_int():
    input_ = ["a", "b", "c"]
    n = 3
    gold = [("a",), ("b",), ("c",), ("a", "b"), ("b", "c"), ("a", "b", "c")]
    res = n_grams(input_, n)
    assert list(res) == gold


def test_n_grams_tuple():
    input_ = ["a", "b", "c"]
    n = (1, 3)
    gold = [("a",), ("b",), ("c",), ("a", "b", "c")]
    res = n_grams(input_, n)
    assert list(res) == gold


def test_geometric_mean():
    input_ = [0.82, 0.061, 0.22]
    gold = 0.22242765817194177
    res = geometric_mean(input_)
    np.testing.assert_allclose(res, gold)


def test_geometric_mean_whole():
    p = np.random.randint(1, 10, size=np.random.randint(4, 10))
    gold = np.power(np.prod(p), 1.0 / len(p))
    res = geometric_mean(p)
    np.testing.assert_allclose(res, gold)


def test_geometric_mean_zero():
    p = np.random.randint(1, 100, size=np.random.randint(4, 10))
    p[np.random.randint(0, len(p))] = 0
    res = geometric_mean(p)
    assert res == 0.0


def test_geometric_mean_neg():
    p = np.random.randint(1, 100, size=np.random.randint(4, 10))
    p[np.random.randint(0, len(p))] = np.random.randint(3, 30) * -1
    res = geometric_mean(p)
    assert res == 0.0


def test_brevity_penalty_not_applied():
    def test():
        p = np.random.randint(100, 1000)
        g = p - np.random.randint(1, p)
        assert p > g, "Your test is wrong"
        res, _ = brevity_penalty(p, g)
        assert res == 1.0

    for _ in range(100):
        test()


def test_brevity_penalty_value():
    p = np.random.randint(100, 1000)
    g = p + np.random.randint(100, 200)
    gold = np.exp(1 - (g / p))
    res, _ = brevity_penalty(p, g)
    np.testing.assert_allclose(res, gold)


def test_brevity_penalty_ratio():
    p = np.random.randint(100, 1000)
    g = np.random.randint(100, 1000)
    gold = p / float(g)
    _, res = brevity_penalty(p, g)
    assert res == gold


def test_count_matches():
    a = Counter()
    b = Counter()
    total = 0
    upper = np.random.randint(3, 6)
    gold = np.zeros(upper)
    for i in range(np.random.randint(10, 20)):
        key = tuple([i] * np.random.randint(1, upper))
        if np.random.rand() > 0.5:
            a_v = np.random.randint(5, 10)
            b_v = np.random.randint(5, 10)
            a[key] = a_v
            b[key] = b_v
            gold[len(key) - 1] += min(a_v, b_v)
        else:
            c = a if np.random.rand() > 0.5 else b
            c[key] = np.random.randint(5, 10)
    res = count_matches(a, b, np.zeros(upper))
    np.testing.assert_equal(res, gold)


def test_count_possible():
    input_ = ["a", "b", "c"]
    n = 3
    gold = [3, 2, 1]
    res = count_possible(input_, np.zeros(n))
    np.testing.assert_equal(res, np.array(gold))


def test_read_lines():
    gold = [[random_str() for _ in range(np.random.randint(1, 100))] for _ in range(np.random.randint(25, 40))]
    input_ = [" ".join(g) for g in gold]
    res = _read_lines(input_, False)
    assert res == gold


def test_read_lines_lowercase():
    gold = [[random_str() for _ in range(np.random.randint(1, 100))] for _ in range(np.random.randint(25, 40))]
    input_ = [" ".join(g) for g in gold]
    gold = [[x.lower() for x in g] for g in gold]
    res = _read_lines(input_, True)
    assert res == gold


def test_read_references_group_references():
    input1 = [
        " ".join([random_str() for _ in range(np.random.randint(1, 100))]) for _ in range(np.random.randint(25, 40))
    ]
    input2 = [" ".join([random_str() for _ in range(np.random.randint(1, 100))]) for _ in range(len(input1))]
    gold = [(input1[i], input2[i]) for i in range(len(input1))]
    with patch("eight_mile.bleu._read_lines") as read_patch:
        with patch("eight_mile.bleu.open"):
            read_patch.side_effect = (input1, input2)
            res = _read_references(["", ""], False)
            assert res == gold
