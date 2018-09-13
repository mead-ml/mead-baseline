from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import numpy as np
from hpctl.sample import random_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', '-t', default=1000, type=int)
    args = parser.parse_args()

    runs = []
    for _ in range(args.trials):
        found = set()
        name = None
        while name not in found:
            found.add(name)
            name = random_name()
        runs.append(len(found) - 1)

    print("{} \u00B1 {} names sampled before a collision.".format(np.mean(runs), np.std(runs)))
    print("Min: {}, Max: {}".format(np.min(runs), np.max(runs)))


if __name__ == "__main__":
    main()
