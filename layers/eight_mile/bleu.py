r"""Multi-bleu in python matching Moses http://www.statmt.org/moses/?n=Moses.SupportTools#ntoc5

Reference files and prediction files must be sentence aligned. To use multiple
references use multiple reference files (all sentence aligned)

All text should already be tokenized (so that calling `.split()` on the line produces correct tokens)

Slight differences between this and multi-ble.pl:

 * multi-bleu.pl crashes when the hypothesis corpus is completely empty. We don't want training to crash
   so we set the brevity penalty to `0` in this case. This makes sense because as the reference corpus grows
   arbitrary large the length ratio (gold / pred) approaches infinity and the limit of e^(1 - x) as
   x -> infinity is 0. When the hypothesis corpus is we also return a np.nan for the length ratio for a
   visual cue that the lengths were weird when using the cli tool.

 * If your reference file doesn't have a newline at the end of the file but your hypothesis does
   then multi-blue.pl will give include this newline as an empty example and your score will
   be lower than it should be. This code will ignore that line and give the correct score.

   For example:
   ```
       $ diff ref pred
       1000c1000
       < 52 95 73 83 93 87 88 81 93
       \ No newline at end of file
       ---
       > 52 95 73 83 93 87 88 81 93

       $ diff ref pred_no_newline
       $

       $ bleu ref < pred
       BLEU = 100.00, 100.0/100.0/100.0/100.0 (BP=1.000, ratio=1.000, hyp_len=7533, ref_len=7533)

       $ bleu ref < pred_no_newline
       BLEU = 100.00, 100.0/100.0/100.0/100.0 (BP=1.000, ratio=1.000, hyp_len=7533, ref_len=7533)

       $ perl multi-bleu.pl ref < pred
       BLEU = 99.98, 100.0/100.0/100.0/100.0 (BP=1.000, ratio=1.000, hyp_len=7533, ref_len=7533)

       $ perl multi-bleu.pl ref < pred_no_newline
       BLEU = 100.00, 100.0/100.0/100.0/100.0 (BP=1.000, ratio=1.000, hyp_len=7533, ref_len=7533)
    ```
"""
import sys
import argparse
from operator import or_
from itertools import chain
from functools import reduce
from collections import Counter
from typing import List, Tuple, Union, Iterable, TextIO, Counter as CounterType, NamedTuple
import numpy as np


__all__ = ["Bleu", "bleu"]


Hypothesis = List[str]
# This is [B, T] where B is the number of hypotheses and T is the number of words in that hypothesis.
HypothesisCorpus = List[Hypothesis]

Reference = List[str]
References = List[Reference]
# This is [B, R, T] where B is the number of references, R is the number of gold references we have
# for a particular example, and T is the number of words in that reference.
ReferenceCorpus = List[References]


class Bleu(NamedTuple):
    """A collection of the information returned from the bleu calculation"""

    bleu: float
    precision: np.ndarray
    brevity_penalty: float
    length_ratio: float
    pred_length: int
    gold_length: int


def n_grams(tokens: List[str], n: Union[int, Tuple[int]]) -> Iterable[Tuple[str]]:
    """Create a list for n grams for each value up to and including n.

    :param tokens: The sequence to create n_grams on.
    :param n: If an int the largest n_gram to create otherwise the size of
        n_grams to create.

    :returns: The n_grams ordered by size.
    """
    n = range(n + 1) if isinstance(n, int) else n
    return chain(*(zip(*[tokens[i:] for i in range(n_)]) for n_ in n))


def count_n_grams(tokens: List[str], n: Union[int, Tuple[int]]) -> CounterType[Tuple[str]]:
    """Count the n_grams in a sequence.

    :param tokens: The sequence to create n_grams on.
    :param n: If an int the largest n_gram to create otherwise the size of
        n_grams to create.

    :returns: The counts of each n_gram.
    """
    return Counter(n_grams(tokens, n))


def find_closest(pred_len: int, golds: References) -> int:
    """Find the gold sentence that has the most similar length to pred.

    Note:
        When there are multiple sentence with the same difference in length
        to the pred length the shortest one is selected.

    :param pred_len: The length of the predicted sentence.
    :param golds: The gold sentences.

    :returns: The length of the selected gold sentence.
    """
    best_diff = sys.maxsize
    best_len = sys.maxsize
    for gold in golds:
        gold_len = len(gold)
        diff = abs(pred_len - gold_len)
        if diff < best_diff:
            best_diff = diff
            best_len = gold_len
        elif diff == best_diff:
            best_len = gold_len if gold_len < best_len else best_len
    return best_len


def corpora_lengths(preds: HypothesisCorpus, golds: ReferenceCorpus) -> Tuple[int, int]:
    """Calculate the length of the two corpora.

    The length of the pred corpus is just the sum of lengths, the length
    of the gold corpus is the sum of the lengths of a single gold selected
    from a set of golds. This single gold is selected to be the length that
    is closest to the pred lengths (in the event of ties select the shorter
    gold sentence).

    :param preds: A list of sentences generated by the model.
    :param golds: A list of gold references. A gold reference can have multiple gold sentences

    :returns: The length of the predicted corpus and the length of the gold corpus.
    """
    pred_len = 0
    gold_len = 0
    for pred, gold in zip(preds, golds):
        pred_len += len(pred)
        gold_len += find_closest(len(pred), gold)
    return pred_len, gold_len


def max_gold_n_gram_counts(golds: References, n: int) -> CounterType[Tuple[str]]:
    """Find the maximum n gram count for the gold sequences.

    :param golds: The gold sentences.
    :param n: The max size of n-grams we are using.

    :returns: The maximum count for any n-gram that appears in a gold sentence.
    """
    # or_ is the union of counters so result is max count for each n gram
    # that appears in the any of the gold sentences.
    return reduce(or_, map(lambda g: count_n_grams(g, n), golds))


def count_matches(
    pred_counts: CounterType[Tuple[str]], gold_counts: CounterType[Tuple[str]], matches: np.ndarray
) -> np.ndarray:
    """Aggregate the number of matches for each n gram length.

    :param pred_counts: The counts for n grams found in the predicted sentence.
    :param gold_counts: The max counts for n grams found in the gold sentences.
    :param matches: The number of matches found so far grouped by n-gram size.

    :returns: The number of matches so far grouped by n-gram size.
    """
    # & is the intersection of counters, selecting the min of each n gram
    # that appears in both of the counters.
    overlap = pred_counts & gold_counts
    for n_gram, count in overlap.items():
        matches[len(n_gram) - 1] += count
    return matches


def count_possible(pred: Hypothesis, total: np.ndarray) -> np.ndarray:
    """Count the total number of possible matches.

    We recalculate the total possible rather than use the counts calculated
    before because the counts aren't binned by length.

    :param pred: The predicted sentence.
    :param total: The current total number of possible matches grouped by size.

    :returns: The new number of possible n-gram matches of far grouped by size.
    """
    for n in range(len(total)):
        # Possible number of n_grams for a sequence is the length of the sequence
        # minus the length of the n_gram plus 1. In this case n is already
        # one less than length of n_gram so just subtract that.
        total_grams = len(pred) - n
        if total_grams > 0:
            total[n] += total_grams
    return total


def geometric_mean(precision: np.ndarray) -> float:
    """Calculate the geometric mean of the precision.

    Geometric mean is the nth root of the product of n numbers. This
    can be expressed as the e^(arithmetic mean of the logs). The geometric
    mean only applies to positive number and any 0 is the values makes the
    answer trivially zero to checkout for that first to avoid log(0).

    :param precision: The precision for each n-gram size.

    :returns: The geometric_mean of the values.
    """
    if np.min(precision) <= 0:
        return 0.0
    return np.exp(np.mean(np.log(precision))).item()


def brevity_penalty(pred_len: int, gold_len: int) -> Tuple[float, float]:
    """Calculate the brevity penalty.

    Note:
        multi-bleu.pl crashes when the hypothesis corpus is completely empty.
        We don't want training to crash then there are no hypotheses so we set
        the brevity penalty to `0` because as the reference corpus grows arbitrary
        large the length ration (gold / pred) approaches infinity and the limit of
        e^(1 - x) as x -> infinity is 0.

        We also return a np.nan for the length ratio for a visual cue that the
        lengths were weird when using the cli tool.

    :param pred_len: The length of the model prediction corpus
    :param gold_len: The length of the gold corpus

    :returns: The brevity penalty and the ratio of predicted length to gold length.
    """
    if pred_len == 0:
        return 0, np.nan
    ratio = pred_len / float(gold_len)
    # If ratio is <= 1.0 then pred_len <= gold_len so penalty applies.
    # Penalty is defined as e^(1 - (gold / pred)). (1 / (p / g)) = (g / p)
    bp = np.exp(1 - (1.0 / ratio)).item() if ratio <= 1.0 else 1.0
    return bp, ratio


def bleu(preds: HypothesisCorpus, golds: ReferenceCorpus, n: int = 4) -> Bleu:
    """Calculate BLEU score

    This implementation is designed to match the output of `multi-bleu,pl` from
    http://www.statmt.org/moses/?n=Moses.SupportTools#ntoc5

    :param preds: A list of sentences generated by the model.
    :param golds: A list of gold references where each reference can contain multiple sentences.
    :param n: The max size n-gram to use.

    :returns: (
        bleu score,
        precision per n_gram size,
        brevity penalty,
        ration of prediction length to reference lengths,
        length of the predictions,
        length of the references
    )
    """
    matches = np.zeros(n)
    total = np.zeros(n)
    pred_len, gold_len = corpora_lengths(preds, golds)
    for pred, gold in zip(preds, golds):
        max_gold_counts = max_gold_n_gram_counts(gold, n)
        pred_counts = count_n_grams(pred, n)
        matches = count_matches(pred_counts, max_gold_counts, matches)
        total = count_possible(pred, total)

    precision = np.array([matches[i] / float(total[i]) if total[i] > 0 else 0.0 for i in range(n)])
    geo_mean = geometric_mean(precision)
    bp, len_ratio = brevity_penalty(pred_len, gold_len)
    b = geo_mean * bp * 100
    return Bleu(b, precision * 100, bp, len_ratio, pred_len, gold_len)


def _read_references(reference_files: List[str], lc: bool) -> ReferenceCorpus:
    """Read from multiple reference files.

    :param reference_file: A list of reference files to read from.
    :param lc: Should the input be lowercased?

    :return: The reference corpus.
    """
    references: List[HypothesisCorpus] = []
    for ref in reference_files:
        with open(ref) as f:
            references.append(_read_lines(f, lc))
    return list(zip(*references))


# Slightly strange file reader that read from pre opened files to facilitate
# reading the predictions from stdin. Won't be used elsewhere.
def _read_lines(f: TextIO, lc: bool) -> HypothesisCorpus:
    """Read from a file that is open.

    The expected file format is each line contains a single example and each example
    is a list of tokens joined on whitespace

    :param f: The file to read from.
    :param lc: Should the input be lowercased?

    :return: The list of examples
    """
    if lc:
        return list(map(lambda x: list(map(lambda y: y.lower(), x.split())), f))
    return list(map(lambda x: x.split(), f))


def main():
    """Use as a cli tools like multibleu.pl"""
    usage = "%(prog)s [-lc] [-n] reference < hypothesis\nReads the references from reference or reference0, reference1, ...\n"
    parser = argparse.ArgumentParser(description="Calculate Bleu score.", usage=usage)
    parser.add_argument("-lc", help="lowercase the input", action="store_true")
    parser.add_argument("-n", type=int, default=4, help="The number of ngrams to use.")
    parser.add_argument("reference", nargs="+")
    args = parser.parse_args()

    golds: ReferenceCorpus = _read_references(args.reference, args.lc)
    preds: HypothesisCorpus = _read_lines(sys.stdin, args.lc)

    b = bleu(preds, golds, args.n)
    precision_str = "/".join(["{:.1f}"] * len(b.precision)).format(*b.precision)

    print(
        f"BLEU = {b.bleu:.2f}, {precision_str} (BP={b.brevity_penalty:.3f}, ratio={b.length_ratio:.3f}, hyp_len={b.pred_length}, ref_len={b.gold_length})"
    )


if __name__ == "__main__":
    main()
