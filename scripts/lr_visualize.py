import os
import math
import json
import inspect
import argparse
from hashlib import sha1
from collections import defaultdict
import numpy as np
import baseline as bl
import matplotlib.pyplot as plt
from lr_compare import plot_learning_rates


# Collect possible Schedulers
OPTIONS = {}
WARMUP = {}
base_classes = (
    bl.train.LearningRateScheduler.__name__,
    bl.train.WarmupLearningRateScheduler.__name__
)
# Collect all schedulers except base classes and separate out the warmup ones
for item_name in dir(bl.train):
    if item_name in base_classes: continue
    item = getattr(bl.train, item_name)
    try:
        if issubclass(item, bl.train.LearningRateScheduler):
            OPTIONS[item_name] = item
        if issubclass(item, bl.train.WarmupLearningRateScheduler):
            WARMUP[item_name] = item
    except:
        pass
REST = {}
for k, v in OPTIONS.items():
    if k not in WARMUP:
        REST[k] = v

# Collect the args and defaults that the schedulers use.
# This currently misses the `lr` because it is kwargs only as well
# As `values` because that is never a defaulted argument
# (we skip over ones without defaults)
ARGS = defaultdict(list)
DEFAULTS = defaultdict(list)
for k, v in OPTIONS.items():
    az = inspect.getargspec(v).args
    ds = inspect.getargspec(v).defaults
    if az is None or ds is None: continue
    for a, d in zip(reversed(az), reversed(ds)):
        ARGS[a].append(k)
        DEFAULTS[a].append(d)

for a, ds in DEFAULTS.items():
    ds_set = set(ds)
    if len(ds_set) == 1:
        # If the only default values is None set to None.
        if None in ds_set:
            DEFAULTS[a] = None
        # Set to the only default.
        else:
            DEFAULTS[a] = ds_set.pop()
    else:
        # Remove None and grab the last default, no reason arbitrary but
        # consistant
        if None in ds: ds.remove(None)
        DEFAULTS[a] = ds[-1]


new_defaults = defaultdict(lambda: None)
for k, v in DEFAULTS.items():
    new_defaults[k] = v

DEFAULTS = new_defaults


def exp_steps(y, lr, decay_rate, decay_steps):
    """Calculate how many steps are needed to get to some value."""
    return decay_steps * (math.log(y / lr) / math.log(decay_rate))


def inv_steps(y, lr, decay_rate, decay_steps):
    """Calculate how many steps are needed to get to some value."""
    return decay_steps * (((lr / y) - 1.0) / decay_rate)


def get_steps(type_, args, double_warm=False):
    """
    Calculate the number of steps to plot. Most of these calculations are
    bespoke to the scheduler.
    """
    if type_ == 'ConstantScheduler':
        return 1000
    # WarmupLinear
    if type_ in WARMUP:
        # If we are just warm up double it to show its fine
        return args['warmup_steps'] * 2 if double_warm else args['warmup_steps']
    # PiecewiseDecay, Zaremba
    if args['boundaries'] is not None:
        # If we are bound based how bounds plus some extra
        gap = np.max(np.diff(args['boundaries']))
        return args['boundaries'][-1] + gap
    if type_ == 'CyclicLRScheduler':
        # Cyclic, show some cycles
        return args['decay_steps'] * args['cycles']
    # Show it decay once
    # Cosine, InverseTimeDecay, Exponential
    if type_ == 'ExponentialDecayScheduler':
        return exp_steps(args['min_lr'], args['lr'], args['decay_rate'], args['decay_steps'])
    if type_ == 'InverseTimeDecayScheduler':
        return inv_steps(args['min_lr'], args['lr'], args['decay_rate'], args['decay_steps'])
    if type_ == 'CosineDecayScheduler':
        return args['decay_steps']


# def plot_learning_rates(ys, names):
#     fig, ax = plt.subplots(1, 1)
#     ax.set_title('Learning Rates from baseline.')
#     ax.set_xlabel('Steps')
#     ax.set_ylabel('Learning Rates')
#     for y, name in zip(ys, names):
#         ax.plot(np.arange(len(y)), y, label=name)
#     ax.legend()
#     return fig


def main():
    parser = argparse.ArgumentParser(description="Plot a learning rate schedule.")
    # Begin lr scheduler based arguments
    parser.add_argument(
        "type",
        choices=OPTIONS.keys(),
        help="The scheduler to visualize."
    )
    parser.add_argument(
        "--lr",
        type=float, default=1.0,
        help="The initial Learning Rate."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int, default=DEFAULTS['warmup_steps'],
        help="The number of steps for a warmup. Used in {}.".format(", ".join(ARGS['warmup_steps']))
    )
    parser.add_argument(
        "--max_lr",
        type=float, default=DEFAULTS['max_lr'],
        help="The maximum learning rate for a cyclical one. Used in {}.".format(", ".join(ARGS['max_lr']))
    )
    parser.add_argument(
        "--decay_steps",
        type=int, default=DEFAULTS['decay_steps'],
        help="The number of steps to take before a decay. Used in {}.".format(", ".join(ARGS['decay_steps']))
    )
    parser.add_argument("--boundaries", nargs="+", default=DEFAULTS['boundaries'])
    parser.add_argument("--values", nargs="+", default=DEFAULTS['values'])
    parser.add_argument(
        "--decay_rate",
        type=float, default=DEFAULTS['decay_rate'],
        help="How much to decay. Used in {}.".format(", ".join(ARGS['decay_rate']))
    )
    parser.add_argument(
        "--staircase",
        action="store_true", default=DEFAULTS['staircase'],
        help="Should you decay in a stepwise fashion? Used in {}.".format(", ".join(ARGS['staircase']))
    )
    parser.add_argument(
        "--alpha",
        type=float, default=DEFAULTS['alpha'],
        help="Alpha. Used in {}.".format(", ".join(ARGS['alpha']))
    )
    parser.add_argument(
        "--warm",
        choices=WARMUP.keys(),
        default=DEFAULTS['warm'],
        help="The Warmup Scheduler to use. Used in {}.".format(", ".join(ARGS['warm']))
    )
    parser.add_argument(
        "--rest",
        choices=REST.keys(),
        default=DEFAULTS['rest'],
        help="The Scheduler to use after warmup. Used in {}.".format(", ".join(ARGS['rest']))
    )
    # Begin Visualization only arguments
    parser.add_argument(
        "--steps", type=int,
        help="Override the number of steps to plot."
    )
    parser.add_argument(
        "--cycles",
        type=int, default=6,
        help="Override the number of cycles to plot."
    )
    parser.add_argument(
        "--min_lr",
        type=float, default=1e-3,
        help="When calculating the number of steps to show how small should the learning rate get before we stop."
    )
    parser.add_argument(
        "--out_file", default='.lrs/cache.index', help="Where to save the results for later plotting."
    )
    args = parser.parse_args()
    args = vars(args)

    # Build the sub schedulers for the Composite Scheduler
    if args['type'] == 'CompositeLRScheduler':
        if args['warm'] is None or args['rest'] is None:
            raise RuntimeError("Warmup and Rest Scheduler are required when the Scheduler is CompositeLRScheduler")
        args['warm'] = WARMUP[args['warm']](**args)
        args['rest'] = REST[args['rest']](**args)

    if args['boundaries'] is not None:
        if args['values'] is not None:
            if len(args['boundaries']) != len(args['values']):
                raise RuntimeError("Bounds and Value list must be aligned")

    lr = OPTIONS[args['type']](**args)

    # Calculate the number of steps you should show.
    if args['steps'] is None:
        if args['type'] == 'CompositeLRScheduler':
            warm = get_steps(type(args['warm']).__name__, args, double_warm=False)
            rest = get_steps(type(args['rest']).__name__, args)
            args['steps'] = warm + rest
        else:
            args['steps'] = get_steps(args['type'], args, double_warm=True)


    # Plot the schedule
    steps = np.arange(0, args['steps'])
    ys = np.stack(lr(s) for s in steps)
    fig = plot_learning_rates([ys], [str(lr)])
    plt.show()

    # Save the lr values to a cache, this is .lrs/{hash of lr values}.npy and save
    # information about it into the cache index
    if args['out_file'] is not None:
        dir_ = os.path.dirname(args['out_file'])
        try: os.makedirs(dir_)
        except: pass
        with open(args['out_file'], 'a') as f:
            lr_hash = sha1(ys.tostring()).hexdigest()
            file_name = "{}.npy".format(lr_hash)
            index = {'name': str(lr), 'file': file_name}
            f.write(json.dumps(index) + "\n")
            np.save(os.path.join(dir_, file_name), ys)

if __name__ == "__main__":
    main()
