import re
import six.moves


class Progress(object):
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


# Modifed from here
# http://stackoverflow.com/questions/3160699/python-progress-bar#3160819
class ProgressBar(Progress):
    """Simple terminal-based progress bar
    
    Writes a progress bar to the terminal, using a designated `symbol` (which defaults to `=`)
    
    """
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='='):
        super(ProgressBar, self).__init__()
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def update(self, step=1):
        """Update progress bar by number of `steps`
        
        :param step: The number of tasks completed
        :return: 
        """
        self.current += step
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        six.print_('\r' + self.fmt % args, end='')

    def done(self):
        """All tasks are finished, complete the progress bar
        
        :return: 
        """
        self.current = self.total
        self.update(step=0)
        print('')

