from baseline.utils import convert_bio_to_iobes
import os
import argparse


parser = argparse.ArgumentParser(description='Translate input sequence to output sequence')
parser.add_argument('--io_dir', help='Input/Output dir', default='../data')
parser.add_argument('--train_file', help='Training file relative name', default='eng.train')
parser.add_argument('--valid_file', help='Validation file relative name', default='eng.testa')
parser.add_argument('--test_file', help='Test file relative name', default='eng.testb')
parser.add_argument('--suffix', help='Suffix to append', default='.iobes')
parser.add_argument('--fields', help='fields to convert to iobes if you have multiple features', default=[-1], nargs='+', required=False)

args = parser.parse_args()


convert_bio_to_iobes(os.path.join(args.io_dir, args.train_file),
                     os.path.join(args.io_dir, args.train_file + args.suffix), fields=[int(x) for x in args.fields])
convert_bio_to_iobes(os.path.join(args.io_dir, args.valid_file),
                     os.path.join(args.io_dir, args.valid_file + args.suffix), fields=[int(x) for x in args.fields])
convert_bio_to_iobes(os.path.join(args.io_dir, args.test_file),
                     os.path.join(args.io_dir, args.test_file + args.suffix), fields=[int(x) for x in args.fields])

