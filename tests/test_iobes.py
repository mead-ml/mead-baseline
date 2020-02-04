import random
from mock import patch
from eight_mile.utils import (
    to_chunks,
    to_chunks_iobes,
    convert_iob_to_bio,
    convert_iob_to_iobes,
    convert_bio_to_iob,
    convert_bio_to_iobes,
    convert_iobes_to_bio,
    convert_iobes_to_iob,
)

# Most of these tests follow this general format. A random sequence of entity spans are generated,
# These entities are used to generate a tag sequence in some given format. These generated entities
# are used as gold for the `to_chunks` code. Generating multiple tag sequences from a single
# random set of spans can be used for gold data for the converting functions.


def generate_spans(ents=["X", "Y"], max_span=5, min_span=0, max_spanl=3, min_spanl=1, min_space=0, max_space=1):
    """Generate a series of entity spans. They are be touching each other but don't overlap."""
    i = 0
    n_spans = random.randint(min_span, max_span)
    spans = []
    for _ in range(n_spans):
        ent = random.choice(ents)
        span_length = random.randint(min_spanl, max_spanl)
        span_gap = random.randint(min_space, max_space)
        span_start = i + span_gap
        span_end = span_start + span_length
        i = span_end
        span = "@".join([ent] + list(map(str, range(span_start, span_end))))
        spans.append(span)
    end = random.randint(min_space, max_space)
    return spans, i + end


def parse_span(span):
    parts = span.split("@")
    return parts[0], list(map(int, parts[1:]))


def generate_iobes(spans):
    """Convert a set of spans to an IOBES sequence."""
    spans, length = spans
    text = ["O"] * length
    for span in spans:
        ent, locs = parse_span(span)
        if len(locs) == 1:
            text[locs[0]] = "S-{}".format(ent)
            continue
        for i, loc in enumerate(locs):
            if i == 0:
                text[loc] = "B-{}".format(ent)
            elif i == len(locs) - 1:
                text[loc] = "E-{}".format(ent)
            else:
                text[loc] = "I-{}".format(ent)
    return text


def generate_bio(spans):
    """Convert a set of spans to a BIO sequence."""
    spans, length = spans
    text = ["O"] * length
    for span in spans:
        ent, locs = parse_span(span)
        for i, loc in enumerate(locs):
            if i == 0:
                text[loc] = "B-{}".format(ent)
            else:
                text[loc] = "I-{}".format(ent)
    return text


def generate_iob(spans):
    """Convert a set of spans to an IOB sequence."""
    spans, length = spans
    text = ["O"] * length
    for span in spans:
        ent, locs = parse_span(span)
        for i, loc in enumerate(locs):
            text[loc] = "I-{}".format(ent)
    for span in spans:
        ent, locs = parse_span(span)
        if locs[0] != 0 and text[locs[0] - 1][2:] == ent:
            text[locs[0]] = "B-{}".format(ent)
    return text


def test_iob_bio():
    def test():
        spans = generate_spans()
        iob = generate_iob(spans)
        gold_bio = generate_bio(spans)
        bio = convert_iob_to_bio(iob)
        assert bio == gold_bio

    for _ in range(100):
        test()


def test_bio_iob():
    def test():
        spans = generate_spans()
        bio = generate_bio(spans)
        gold_iob = generate_iob()
        iob = convert_bio_to_iob(bio)
        assert iob == gold_iob


def test_bio_iobes():
    def test():
        spans = generate_spans()
        bio = generate_bio(spans)
        gold_iobes = generate_iobes(spans)
        iobes = convert_bio_to_iobes(bio)
        assert iobes == gold_iobes

    for _ in range(100):
        test()


def test_iobes_bio():
    def test():
        spans = generate_spans()
        iobes = generate_iobes(spans)
        gold_bio = generate_bio(spans)
        bio = convert_iobes_to_bio(iobes)
        assert bio == gold_bio

    for _ in range(100):
        test()


def test_iobes_iob():
    def test():
        spans = generate_spans()
        iobes = generate_iobes(spans)
        gold_iob = generate_iob(spans)
        iob = convert_iobes_to_iob(iobes)
        assert iob == gold_iob


def test_iob_bio_cycle():
    def test():
        spans = generate_spans()
        gold_iob = generate_iob(spans)
        res = convert_bio_to_iob(convert_iob_to_bio(gold_iob))
        assert res == gold_iob

    for _ in range(100):
        test()


def test_bio_iob_cycle():
    def test():
        spans = generate_spans()
        gold_bio = generate_bio(spans)
        res = convert_iob_to_bio(convert_bio_to_iob(gold_bio))
        assert res == gold_bio

    for _ in range(100):
        test()


def test_iobes_bio_cycle():
    def test():
        spans = generate_spans()
        gold_iobes = generate_iobes(spans)
        res = convert_bio_to_iobes(convert_iobes_to_bio(gold_iobes))
        assert res == gold_iobes

    for _ in range(100):
        test()


def test_bio_iobes_cycle():
    def test():
        spans = generate_spans()
        gold_bio = generate_bio(spans)
        res = convert_iobes_to_bio(convert_bio_to_iobes(gold_bio))
        assert res == gold_bio

    for _ in range(100):
        test()


def test_iobes_iob_cycle():
    def test():
        spans = generate_spans()
        gold_iobes = generate_iobes(spans)
        res = convert_iob_to_iobes(convert_iobes_to_iob(gold_iobes))
        assert res == gold_iobes

    for _ in range(100):
        test()


def test_iob_iobes_cycle():
    def test():
        spans = generate_spans()
        gold_iob = generate_iob(spans)
        res = convert_iobes_to_iob(convert_iob_to_iobes(gold_iob))
        assert res == gold_iob

    for _ in range(100):
        test()


def test_to_chunks_iobes():
    def test():
        spans = generate_spans()
        gold = spans[0]
        iobes = generate_iobes(spans)
        chunks = to_chunks_iobes(iobes)
        assert chunks == gold

    for _ in range(100):
        test()


def test_to_chunks_iobes_delim():
    def test():
        delim = random.choice(["@", "#", "%"])
        spans = generate_spans()
        gold = spans[0]
        gold = [g.replace("@", delim) for g in gold]
        iobes = generate_iobes(spans)
        chunks = to_chunks_iobes(iobes, delim=delim)
        assert chunks == gold

    for _ in range(100):
        test()


def test_to_chunks_calls_iobes():
    seq = [random.randint(0, 5) for _ in range(random.randint(1, 6))]
    verbose = random.choice([True, False])
    delim = random.choice(["@", "#", "%"])
    with patch("eight_mile.utils.to_chunks_iobes") as iobes_patch:
        to_chunks(seq, "iobes", verbose, delim)
    iobes_patch.assert_called_once_with(seq, verbose, delim)


def test_to_chunks_bio():
    def test():
        spans = generate_spans()
        gold = spans[0]
        iobes = generate_bio(spans)
        chunks = to_chunks(iobes, "bio")
        assert chunks == gold

    for _ in range(100):
        test()


def test_to_chunks_bio_delim():
    def test():
        delim = random.choice(["@", "#", "!"])
        spans = generate_spans()
        gold = spans[0]
        gold = [g.replace("@", delim) for g in gold]
        iobes = generate_bio(spans)
        chunks = to_chunks(iobes, "bio", delim=delim)
        assert chunks == gold

    for _ in range(100):
        test()


def test_to_chunks_iob():
    def test():
        spans = generate_spans()
        gold = spans[0]
        iobes = generate_iob(spans)
        chunks = to_chunks(iobes, "iob")
        assert chunks == gold

    for _ in range(100):
        test()


def test_to_chunks_iob_delim():
    def test():
        delim = random.choice(["@", "#", "$"])
        spans = generate_spans()
        gold = spans[0]
        gold = [g.replace("@", delim) for g in gold]
        iobes = generate_iob(spans)
        chunks = to_chunks(iobes, "iob", delim=delim)
        assert chunks == gold

    for _ in range(100):
        test()


# Tests By Hand
def test_iob_bio_i_after_o():
    in_ = ["O", "I-X", "O"]
    gold = ["O", "B-X", "O"]
    res = convert_iob_to_bio(in_)
    assert res == gold


def test_bio_iob_b_after_diff():
    in_ = ["O", "B-X", "B-Y", "O"]
    gold = ["O", "I-X", "I-Y", "O"]
    res = convert_bio_to_iob(in_)
    assert res == gold


def test_bio_iob_b_after_same():
    in_ = ["O", "B-X", "B-X", "O"]
    gold = ["O", "I-X", "B-X", "O"]
    res = convert_bio_to_iob(in_)
    assert res == gold


def test_iob_bio_i_after_b_diff():
    in_ = ["O", "B-X", "I-Y", "O"]
    gold = ["O", "B-X", "B-Y", "O"]
    res = convert_iob_to_bio(in_)
    assert res == gold


def test_iob_bio_i_after_b_same():
    in_ = ["O", "B-X", "I-X", "O"]
    gold = ["O", "B-X", "I-X", "O"]
    res = convert_iob_to_bio(in_)
    assert res == gold


def test_iob_bio_i_after_i_diff():
    in_ = ["O", "I-X", "I-Y", "O"]
    gold = ["O", "B-X", "B-Y", "O"]
    res = convert_iob_to_bio(in_)
    assert res == gold


def test_iob_bio_i_after_i_same():
    in_ = ["O", "I-X", "I-X", "O"]
    gold = ["O", "B-X", "I-X", "O"]
    res = convert_iob_to_bio(in_)
    assert res == gold


def test_iob_bio_i_first():
    in_ = ["I-X", "O"]
    gold = ["B-X", "O"]
    res = convert_iob_to_bio(in_)
    assert res == gold


def test_bio_iobes_single_b():
    in_ = ["O", "B-X", "O"]
    gold = ["O", "S-X", "O"]
    res = convert_bio_to_iobes(in_)
    assert res == gold


def test_bio_iobes_i_to_e_at_end():
    in_ = ["O", "B-X", "I-X"]
    gold = ["O", "B-X", "E-X"]
    res = convert_bio_to_iobes(in_)
    assert res == gold


def test_bio_iobes_i_to_e_at_o():
    in_ = ["O", "B-X", "I-X", "O"]
    gold = ["O", "B-X", "E-X", "O"]
    res = convert_bio_to_iobes(in_)
    assert res == gold


def test_bio_iobes_i_to_e_at_b():
    in_ = ["O", "B-X", "I-X", "B-Y", "I-Y"]
    gold = ["O", "B-X", "E-X", "B-Y", "E-Y"]
    res = convert_bio_to_iobes(in_)
    assert res == gold
