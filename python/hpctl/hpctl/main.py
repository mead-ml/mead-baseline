from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from mead.utils import convert_path
from hpctl.utils import hpctl_path
from hpctl.core import search, find, list_names, serve, launch, verify

import warnings
warnings.simplefilter("ignore")


def add_common_args(p):
    p.add_argument(
        '--config', type=hpctl_path,
        help="Configuration for an experiment."
    )
    p.add_argument(
        '--settings',
        default=None, type=convert_path,
        help="Configuration."
    )
    p.add_argument(
        '--datasets',
        default='config/datasets.json', type=convert_path,
        help="Library of dataset labels."
    )
    p.add_argument(
        '--embeddings',
        default='config/embeddings.json', type=convert_path,
        help="Library of embedding labels."
    )
    p.add_argument(
        '--logging',
        default='config/logging.json', type=convert_path,
        help="Configuration for mead logging."
    )
    p.add_argument(
        '--hpctl-logging',
        default='config/hpctl-logging.json', type=hpctl_path,
        help="Configuration for hpctl logging."
    )
    p.add_argument(
        '--task',
        choices=['classify', 'tagger', 'seq2seq', 'lm'],
        help="Task to run."
    )
    p.add_argument('--reporting', help='reporting hooks', nargs='+')
    p.add_argument('--label', help="The name for the job in xpctl.")
    p.add_argument('--num_iters', type=int, help="The number of sample to run.", default=2)


def print_help(p):
    def ph(*args, **kwargs):
        p.print_help()
    return ph


def main():
    parser = argparse.ArgumentParser(
        prog="hpctl",
        description="Hyper Parameter Optimization tools.",
    )
    ph = print_help(parser)
    parser.set_defaults(func=ph)

    subparsers = parser.add_subparsers()

    search_parser = subparsers.add_parser('search', description="Explore Hyper Parameters.")
    search_parser.set_defaults(func=search)
    add_common_args(search_parser)

    verify_parser = subparsers.add_parser('verify', description="Run a config a lot.")
    verify_parser.set_defaults(func=verify)
    add_common_args(verify_parser)

    launch_parser = subparsers.add_parser('launch')
    launch_parser.set_defaults(func=launch)
    add_common_args(launch_parser)
    launch_parser.set_defaults(num_iters=1)

    find_parser = subparsers.add_parser('find', description="Find a file from the human name.")
    find_parser.set_defaults(func=find)
    find_parser.add_argument("name")
    find_parser.add_argument("--root", default="delete_me")

    list_parser = subparsers.add_parser('list', description="List all names of experiment results.")
    list_parser.add_argument("--root", default="delete_me")
    list_parser.set_defaults(func=list_names)

    serve_parser = subparsers.add_parser('serve', description="Start a flask server.")
    serve_parser.set_defaults(func=serve)
    add_common_args(serve_parser)
    serve_parser.add_argument("--debug", action="store_true")

    args, extra = parser.parse_known_args()
    args = vars(args)
    args['unknown'] = extra
    args.pop('func')(**args)


if __name__ == "__main__":
    main()
