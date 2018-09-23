from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves.queue import Empty
from six.moves import zip, map, range

import sys
import select
import platform
from multiprocessing import Process, Queue
from baseline.utils import export as exporter
from hpctl.utils import Label
from hpctl.results import States


__all__ = []
export = exporter(__all__)


def color(state, off=False):
    """Turn the state into a colored string.

    :param state: hpctl.results.States, The state to print.
    :param off: bool, Force no color.

    :returns:
        str, The string ready for printing.
    """
    GREEN = '\033[32;1m'
    RED = '\033[31;1m'
    YELLOW = '\033[33;1m'
    BLACK = '\033[30;1m'
    RESTORE = '\033[0m'
    if platform.system() == 'Windows' or off:
        try:
            return str(state).decode('utf-8')
        except:
            return str(state)
    if state is States.DONE:
        color = GREEN
    elif state is States.KILLED:
        color = RED
    elif state is States.RUNNING:
        color = YELLOW
    else:
        color = BLACK
    # Because we use unicode strings for formatting we want the results
    # to be as a unicode string rather than a bytes str, Python 2 returns
    # bytes and 3 returns a str so we need to try to decode it for 2.
    try:
        return "{}{}{}".format(color, str(state).decode('utf-8'), RESTORE)
    except:
        return "{}{}{}".format(color, str(state), RESTORE)


def reset_screen(lines):
    """Update the console.

    LINUX ONLY

    :param lines: int, The number of lines that are to be erased.
    """
    if platform.system() == 'Linux':
        for _ in range(lines):
            print("\033[F \033[2K", end='\r')
    else:
        print()


@export
class Frontend(object):
    """Frontend object that displays information to user and collects input."""

    def __init__(self, *args, **kwargs):
        super(Frontend, self).__init__()

    def update(self):
        """Function that tells the frontend to refresh."""
        pass

    def finalize(self, *args, **kwargs):
        """Function that tells the frontend that no more models will be run."""
        pass

    def command(self):
        """Function that gets any user inputs."""
        pass


class FlaskShim(Frontend):
    """A small class that bridges between the core hpctl and the flask server."""
    def __init__(self, q, fe, **kwargs):
        super(FlaskShim, self).__init__()
        self.queue = q
        self.frontend = fe
        self.frontend.start()

    def finalize(self):
        self.frontend.terminate()

    def command(self):
        try:
            data = self.queue.get(timeout=0.0)
            return data
        except Empty:
            data = None
        return None

@export
class Console(Frontend):
    header = "{0} hpctl search {0}".format("=" * 42)
    """Frontend that prints all ticks to console.

    :param exp: hpctl.experiment.Experiment: The experiment config.
    :param results: hpctl.results.Results: The data storage object.
    :param train: str, The training metric to display.
    :param dev: str, The dev metric to track.
    :param test: str, The test metric to track.
    """
    def __init__(self, exp, results, train=None, dev=None, test=None, **kwargs):
        super(Console, self).__init__()
        self.exp = exp
        self.results = results
        self.print_count = 0
        self.first = True
        default = self.exp.mead_config['train'].get('early_stopping_metric', 'avg_loss')
        self.train_metric = default if train is None else train
        self.dev_metric = default if dev is None else dev
        self.test_metric = default if test is None else test
        print(self.header)

    def update(self):
        """Display the train and dev metrics as they come in."""
        reset_screen(self.print_count)
        self.print_count = 0
        labels = self.results.get_labels(self.exp.experiment_hash)
        max_len = max(map(len, map(lambda x: x.human, labels)))
        for label in labels:
            print('{state} {name:{width}} - train ({train_metric}): {train_stat:.3f} at {train_step} dev ({metric}): {dev_stat:.3f} at {dev_step}'.format(
                state=color(self.results.get_state(label)),
                name=label.human,
                train_metric=self.train_metric,
                train_stat=self.results.get_recent(label, 'Train', self.train_metric),
                train_step=self.results.get_recent(label, 'Train', 'tick'),
                metric=self.dev_metric,
                dev_stat=self.results.get_recent(label, 'Valid', self.dev_metric),
                dev_step=self.results.get_recent(label, 'Valid', 'tick'),
                width=max_len)
            )
            self.print_count += 1

    def finalize(self):
        """Find and print the best results on the test set."""
        self.update()
        best, _, _ = self.results.find_best(self.exp.experiment_hash, 'Valid', self.dev_metric)
        test = self.results.get_recent(best, 'Test', self.test_metric)
        print("\n{} had a test performance of {} on {}".format(best.human, test, self.test_metric))

    def command(self):
        """Function that gets any user inputs.

        Linux only.
        """
        if platform.system() == "Linux":
            r, _, _ = select.select([sys.stdin,], [], [], 0.0)
            if r:
                data = sys.stdin.readline().rstrip("\n")
                self.print_count += 1
                # Simple for now
                data = data.split()
                human, sha1 = self.results.get_label_prefix(data[1])
                if len(data) >= 2:
                    label = Label(
                        self.exp.experiment_hash,
                        sha1[0],
                        human,
                    )
                    data = {
                        "command": data[0],
                        "label": label
                    }
                return data
        return None


@export
class ConsoleDev(Console):
    header = "{0} hpctl search {0}".format("=" * 22)
    """Frontend that prints only the best dev scores to screen.

    :param exp: hpctl.experiment.Experiment: The experiment config.
    :param results: hpctl.results.Results: The data storage object.
    """

    def update(self):
        """Display the best Dev metrics."""
        reset_screen(self.print_count)
        self.print_count = 0
        labels, vals, idxs = self.results.get_best_per_label(self.exp_hash, 'Valid', self.dev_metric)
        max_len = max(map(len, map(lambda x: x.human, human_labels)))
        for label, val, idx in zip(labels, vals, idxs):
            print('{state} {name:{width}} - best dev ({metric}): {stat:.3f} at {step}'.format(
                state=color(self.results.get_state(label)),
                name=label.human, metric=self.dev_metric, stat=val, step=idx,
                width=max_len)
            )
            self.print_count += 1


FRONTENDS = {
    "console": Console,
    "console_dev": ConsoleDev,
    "dummy": Frontend,
    "default": Console,
}


@export
def get_frontend(exp, results):
    """Create a frontend object.

    :param exp: hpctl.experiment.Experiment: The experiment config.
    :param results: hpctl.results.Results: The data storage object.

    :returns:
        hpctl.frontend.Frontend: The frontend object.
    """
    config = exp.frontend_config
    frontend = config.pop('type', 'default')
    if frontend == 'flask':
        from hpctl.flask_frontend import create_flask
        q = Queue()
        fe = create_flask(q, exp, results)
        return FlaskShim(q, fe, **config)

    return FRONTENDS[frontend](exp, results, **config)
