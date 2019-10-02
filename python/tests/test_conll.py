import os
import random
from io import StringIO
from baseline.utils import (
    sniff_conll_file,
    read_conll_docs,
    read_conll_docs_md,
    read_conll_sentences,
    read_conll_sentences_md,
)


file_loc = os.path.realpath(os.path.dirname(__file__))
TEST_FILE = os.path.join(file_loc, 'test_data', 'test.conll')

gold_sentences = [
    [
        ['a', '1', '2'],
        ['b', '3', '4'],
        ['c', '5', '6'],
    ],
    [
        ['d', '7', '8'],
        ['e', '9', '10'],
    ],
    [
        ['g', '11', '12'],
        ['h', '13', '14'],
        ['i', '15', '16'],
        ['j', '17', '18'],
    ],
    [
        ['k', '19', '20'],
    ],
    [
        ['l', '21', '22'],
        ['m', '23', '24'],
        ['n', '25', '26'],
    ],
    [
        ['o', '27', '28'],
        ['p', '29', '30']
    ]
]

gold_documents = [
    [gold_sentences[0], gold_sentences[1]],
    [gold_sentences[2], gold_sentences[3]],
    [gold_sentences[4], gold_sentences[5]]
]

comments = [
    '# begin doc',
    '# This is a comment',
    '# This is the second sentence',
    '# This is an extra comment',
    '# end doc',
    '# begin doc',
    '# 2 2',
    '# begin doc',
]

sentence_comments = [
    [comments[0], comments[1]],
    [comments[2], comments[3]],
    [comments[4], comments[5]],
    [comments[6]],
    [comments[7]],
    []
]

doc_comments = [
    [comments[0]],
    [comments[4], comments[5]],
    [comments[7]],
]

doc_sent_comments = [
    [
        [comments[1]],
        [comments[2], comments[3]],
    ],
    [
        [],
        [comments[6]],
    ],
    [
        [],
        [],
    ]
]

def test_read_conll_sentences():
    for p, g in zip(read_conll_sentences(TEST_FILE), gold_sentences):
        assert p == g


def test_read_conll_sentences_md():
    for (p, pm), g, m in zip(read_conll_sentences_md(TEST_FILE), gold_sentences, sentence_comments):
        assert p == g
        assert pm == m


def test_read_conll_docs():
    for d, g in zip(read_conll_docs(TEST_FILE), gold_documents):
        assert d == g


def test_sniff_conll():
    files = StringIO()
    for _ in range(random.randint(0, 4)):
        files.write("#{}\n".format(random.random()))
    gold = random.randint(2, 12)
    files.write(" ".join(["a"] * gold) + "\n")
    files.seek(0)
    res = sniff_conll_file(files)
    assert res == gold


def test_sniff_reset():
    files = StringIO()
    files.write("We shouldn't see this\n")
    start = files.tell()
    gold = "#This is the gold!"
    files.write(gold + "\n")
    for _ in range(random.randint(0, 3)):
        files.write('#\n')
    files.write("a a\n")
    files.seek(start)
    _ = sniff_conll_file(files)
    res = files.readline().rstrip()
    assert res == gold


def test_read_conll_docs_md():
    for (d, dm, dsm), g, gd, gsm in zip(read_conll_docs_md(TEST_FILE), gold_documents, doc_comments, doc_sent_comments):
        assert d == g
        assert dm == gd
        assert dsm == gsm
