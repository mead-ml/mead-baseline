import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_rates(ys, names):
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Learning Rates from baseline.')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Learning Rates')
    for y, name in zip(ys, names):
        ax.plot(np.arange(len(y)), y, label=name)
    ax.legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare various schedules")
    parser.add_argument("--index", default='.lrs/cache.index', help="The location of the lr index")
    args = parser.parse_args()

    dir_ = os.path.dirname(args.index)
    index = {}
    with open(args.index) as f:
        for line in f:
            line = json.loads(line)
            try:
                index[line['name']] = np.load(os.path.join(dir_, line['file']))
            except:
                pass

    keys = index.keys()

    plot_learning_rates([index[k] for k in keys], keys)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
