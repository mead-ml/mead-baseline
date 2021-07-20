import argparse
from baseline.reader import _norm_ext
from eight_mile.utils import convert_iobes_conll_to_bio


def main():
    parser = argparse.ArgumentParser(description="Convert a CONLL file from IOBES to BIO. This defaults to changing the right two most columns and is used to convert the output of IOBES taggers so that conlleval.pl will work on them.")
    parser.add_argument("--file", help="The file to convert", default="conllresults.conll")
    parser.add_argument("--fields", help="The fields to convert", default=[-2, -1], type=int, nargs="+")
    parser.add_argument("--delim", "-d", help="The delimiter between items", default=" ")
    parser.add_argument("--suffix", help="Suffix to append", default=".bio", type=_norm_ext)
    args = parser.parse_args()

    convert_iobes_conll_to_bio(args.file, args.file + args.suffix, args.fields, args.delim)


if __name__ == '__main__':
    main()