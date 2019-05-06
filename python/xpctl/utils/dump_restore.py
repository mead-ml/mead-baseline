""" dump one xpctl database to local file system, restore back to another database"""
import argparse
import os
from xpctl.backend.core import ExperimentRepo
from baseline.utils import read_config_file


def main():
    parser = argparse.ArgumentParser(description='dump restore xpctl databases')
    parser.add_argument('--from_dbtype', help='backend type, `from` database', default='mongo')
    parser.add_argument('--from_cred', help='credential for backend, `from` database',
                        default=os.path.expanduser('~/xpctlcred.json'))
    parser.add_argument('--to_dbtype', help='backend type, `to` database', default='mongo')
    parser.add_argument('--to_cred', help='credential for backend, `to` database',
                        default=os.path.expanduser('~/xpctlcred-localhost.json'))
    args = parser.parse_args()
    
    d1 = read_config_file(args.from_cred)
    d1.update({'dbtype': args.from_dbtype})
    from_backend = ExperimentRepo().create_repo(**d1)
    dump_file = from_backend.dump()

    d2 = read_config_file(args.to_cred)
    d2.update({'dbtype': args.to_dbtype})
    to_backend = ExperimentRepo().create_repo(**d2)
    to_backend.restore(dump_file)
    

if __name__ == "__main__":
    main()
