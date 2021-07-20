import os
import argparse
from eight_mile.utils import convert_bio_conll_to_iobes

def main():
    parser = argparse.ArgumentParser(description='Convert a CONLL file with BIO tagging to IOBES.')
    parser.add_argument('--io_dir', help='Input/Output dir', default='../data')
    parser.add_argument('--train_file', help='Training file relative name', default='eng.train.bio')
    parser.add_argument('--valid_file', help='Validation file relative name', default='eng.testa.bio')
    parser.add_argument('--test_file', help='Test file relative name', default='eng.testb.bio')
    parser.add_argument('--suffix', help='Suffix to append', default='.iobes')
    parser.add_argument("--fields", help="The fields to convert", default=[-1], type=int, nargs="+")
    parser.add_argument("--delim", help="delimiter for the fields")

    args = parser.parse_args()

    train_output = args.train_file[:-4] if args.train_file.endswith(".bio") else args.train_file
    valid_output = args.valid_file[:-4] if args.valid_file.endswith(".bio") else args.valid_file
    test_output = args.test_file[:-4] if args.test_file.endswith(".bio") else args.test_file

    convert_bio_conll_to_iobes(
        os.path.join(args.io_dir, args.train_file), os.path.join(args.io_dir, train_output + args.suffix),
        fields=args.fields,
        delim=args.delim
    )

    convert_bio_conll_to_iobes(
        os.path.join(args.io_dir, args.valid_file), os.path.join(args.io_dir, valid_output + args.suffix),
        fields=args.fields,
        delim=args.delim
    )

    convert_bio_conll_to_iobes(
        os.path.join(args.io_dir, args.test_file), os.path.join(args.io_dir, test_output + args.suffix),
        fields=args.fields,
        delim=args.delim
    )


if __name__ == '__main__':
    main()
