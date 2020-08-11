"""An implementation of conlleval.pl https://www.clips.uantwerpen.be/conll2000/chunking/output.html

Input is a conll file with the rightmost columns being gold and predicted tags
in that order. Sentences are separated by a blank line.

Some parts of the perl script are not replicated (this doesn't generate a latex
table) but there are other improvements like it supports files in the `IOBES`
format.

The script produces the same output as conlleval.pl for BIO and IOB tagged file.
When running on IOBES files it will produce the same F1 scores as if you
converted the file to BIO and run conlleval.pl however the accuracy will be
slightly different IOBES accuracy will always equal to or lower than BIO score.

This difference is because if the error is on a B-, I-, or O token then the
error will be in both IOBES and BIO. In the conversion from IOBES to BIO then
S- is converted to B- and E- to I- if the IOBES was correct then they will be
converted and will still be correct. If the token was wrong in that is said
it was O or something then it will be wrong after the conversion too. If it was
wrong in that the S- was tagged as B- or the E- was an I- (this is possible and
a common error) then when the gold is changed these will become correct. So it
is only possible that the conversion will make some answers correct. It won't
make anything that was correct wrong.
"""

import sys
import argparse
from itertools import chain
from eight_mile.utils import to_chunks, per_entity_f1, conlleval_output, read_conll_sentences


def _read_conll_file(f, delim):
    """Read a golds and predictions out of a conll file.

    :param f: `file` The open file object.
    :param delim: `str` The symbol that separates columns in the file.

    :returns: `Tuple[List[List[str]], List[List[str]]]` The golds
        and the predictions. They are aligned lists and each element
        is a List of strings that are the list of tags.

    Note:
        the file should contain lines with items separated
        by $delimiter characters (default space). The final
        two items should contain the correct tag and the
        guessed tag in that order. Sentences should be
        separated from each other by empty lines.
    """
    golds = []
    preds = []
    for lines in read_conll_sentences(f, delim=delim):
        golds.append([l[-2] for l in lines])
        preds.append([l[-1] for l in lines])
    return golds, preds


def _get_accuracy(golds, preds):
    """Calculate the token level accuracy.

    :param golds: `List[List[str]]` The list of golds of each example.
    :param preds: `List[List[str]]` The list of predictions of each example.

    :returns: `Tuple[float, int]` The Accuracy and the total number of tokens.
    """
    total = 0
    correct = 0
    for g, p in zip(chain(*golds), chain(*preds)):
        if g == p:
            correct += 1
        total += 1
    return correct / float(total) * 100, total


def _get_entities(golds, preds, span_type="iobes", verbose=False):
    """ Convert the tags into sets of entities.

    :param golds: `List[List[str]]` The list of gold tags.
    :param preds: `List[List[str]]` The list of predicted tags.
    :param span_type: `str` The span labeling scheme used.
    :param verbose: `bool` Should warnings be printed when an illegal transistion is found.
    """
    golds = [set(to_chunks(g, span_type, verbose)) for g in golds]
    preds = [set(to_chunks(p, span_type, verbose)) for p in preds]
    return golds, preds


def main():
    """Use as a cli tool like conlleval.pl"""
    usage = "usage: %(prog)s [--span_type {bio,iobes,iob}] [-d delimiterTag] [-v] < file"
    parser = argparse.ArgumentParser(
        description="Calculate Span level F1 from the CoNLL-2000 shared task.", usage=usage
    )
    parser.add_argument(
        "--span_type",
        default="iobes",
        choices={"iobes", "iob", "bio"},
        help="What tag annotation scheme is this file using.",
    )
    parser.add_argument("--delimiterTag", "-d", help="The separator between items in the file.")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Output warnings when there is an illegal transition."
    )
    args = parser.parse_args()

    golds, preds = _read_conll_file(sys.stdin, args.delimiterTag)
    acc, tokens = _get_accuracy(golds, preds)
    golds, preds = _get_entities(golds, preds, args.span_type, args.verbose)
    metrics = per_entity_f1(golds, preds)
    metrics["acc"] = acc
    metrics["tokens"] = tokens
    print(conlleval_output(metrics))


if __name__ == "__main__":
    main()
