import re
import sys
import six
from eight_mile.utils import exporter, register, optional_params


__all__ = []
export = exporter(__all__)
MEAD_LAYERS_PROGRESS = {}


@export
@optional_params
def register_progress(cls, name=None):
    return register(cls, MEAD_LAYERS_PROGRESS, name, 'progress')


@export
class Progress:
    """Progress hook

    Provide interface for progress updates
    """

    def __init__(self):
        pass

    def update(self, step=1):
        """Update progress by number of `steps`

        :param step: The number of tasks completed
        :return:
        """
        pass

    def done(self):
        """Triggered when all tasks are completed

        :return:
        """
        pass

    def __call__(self, real_iter):
        return self.__iter__(real_iter)

    def __iter__(self, real_iter):
        for x in real_iter:
            yield x
            self.update()
        self.done()


@export
@register_progress('jupyter')
class ProgressBarJupyter(Progress):
    """Simple Jupyter progress bar

    Writes a progress bar to an ipython widget

    """

    def __init__(self, total):
        super().__init__()
        from ipywidgets import FloatProgress
        from IPython.display import display

        self.progress = FloatProgress(min=0, max=total)
        display(self.progress)

    def update(self, step=1):
        self.progress.value += step

    def done(self):
        """Close the widget
        """
        self.progress.close()


# Modifed from here
# http://stackoverflow.com/questions/3160699/python-progress-bar#3160819
@export
@register_progress('terminal')
class ProgressBarTerminal(Progress):
    """Simple terminal-based progress bar

    Writes a progress bar to the terminal, using a designated `symbol` (which defaults to `=`)

    """

    DEFAULT = "Progress: %(bar)s %(percent)3d%%"
    FULL = "%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go"

    def __init__(self, total, width=40, fmt=DEFAULT, symbol="="):
        super().__init__()
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.fmt = re.sub(r"(?P<name>%\(.+?\))d", r"\g<name>%dd" % len(str(total)), fmt)

        self.current = 0
        self.print_ = six.print_ if sys.stdout.isatty() else lambda *args, **kwargs: None
        # Force initial print of pg for when steps are long.
        self.update(step=0)

    def update(self, step=1):
        """Update progress bar by number of `steps`

        :param step: The number of tasks completed
        :return:
        """
        self.current += step
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "remaining": remaining,
        }
        self.print_("\r" + self.fmt % args, end="")

    def done(self):
        """All tasks are finished, complete the progress bar

        :return:
        """
        self.current = self.total
        self.update(step=0)
        self.print_("")

@register_progress('tqdm')
class ProgressBarTQDM(Progress):
    """Simple TQDM progress bar

    Writes a progress bar to an ipython widget

    """

    def __init__(self, total):
        super().__init__()
        from tqdm import tqdm

        self.progress = tqdm(total=total)

    def update(self, step=1):
        self.progress.update(step)

    def done(self):
        """Close the widget
        """
        self.progress.close()


@export
def create_progress_bar(steps, name='default', **kwargs):
    return MEAD_LAYERS_PROGRESS[name](steps, **kwargs)


def SET_DEFAULT_PROGRESS_BAR(type):
    MEAD_LAYERS_PROGRESS['default'] = MEAD_LAYERS_PROGRESS[type]

SET_DEFAULT_PROGRESS_BAR('terminal')
