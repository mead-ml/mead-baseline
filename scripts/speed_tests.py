import argparse
from mead.utils import convert_path
from speed_test.run import run, add
from speed_test.report import report, query, explore


def print_help(p):
    def ph(*args, **kwargs):
        p.print_help()
    return ph


def main():
    parser = argparse.ArgumentParser(prog='Baseline Speed Test.', description="Run speed tests for baseline.")
    ph = print_help(parser)
    parser.set_defaults(func=ph)
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', description='Run a speed test.')
    run_parser.set_defaults(func=run)
    run_parser.add_argument('--config', default='./speed_configs')
    run_parser.add_argument('--frameworks', default=None, nargs='+')
    run_parser.add_argument('--single', action='store_true')
    run_parser.add_argument('--no_crf', action='store_true')
    run_parser.add_argument('--no_attn', action='store_true')
    run_parser.add_argument('--logging', default="config/logging.json", type=convert_path)
    run_parser.add_argument('--datasets', default="config/datasets.json", type=convert_path)
    run_parser.add_argument('--embeddings', default="config/embeddings.json", type=convert_path)
    run_parser.add_argument('--settings', default="config/mead-settings.json", type=convert_path)
    run_parser.add_argument('--db', default='speed.db')
    run_parser.add_argument('--trials', default=5, type=int)
    run_parser.add_argument('--gpu', default=0, type=int)
    run_parser.add_argument('--verbose', action='store_true')

    query_parser = subparsers.add_parser('query', description='Get results about a setup.')
    query_parser.set_defaults(func=query)
    query_parser.add_argument('--db', default='speed.db')
    query_parser.add_argument('--task', default='classify')
    query_parser.add_argument('--dataset', default='SST2')
    query_parser.add_argument('--frameworks', nargs='+')
    query_parser.add_argument('--models', nargs='+')
    query_parser.add_argument('--framework_versions', '-fwv', nargs='+')
    query_parser.add_argument('--baseline_versions', '-blv', nargs='+')

    explore_parser = subparsers.add_parser('explore', description='Explore the database.')
    explore_parser.set_defaults(func=explore)
    explore_parser.add_argument('--db', required=True)

    report_parser = subparsers.add_parser('report', description='Generate a report based on a speed test.')
    report_parser.set_defaults(func=report)
    report_parser.add_argument('--db', default='speed.db')
    report_parser.add_argument('--out', default='speed_test')

    add_parser = subparsers.add_parser('add', description='Add a run to the database')
    add_parser.set_defaults(func=add)
    add_parser.add_argument('--config', required=True)
    add_parser.add_argument('--log', required=True)
    add_parser.add_argument('--db', required=True)
    add_parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
