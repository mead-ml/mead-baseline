from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import reduce

import operator
from collections import defaultdict, deque


class Scheduler(object):
    def __init__(self):
        super(Scheduler, self).__init__()

    def add(self, label, job):
        """Add a job to be scheduled.

        :param label: Label, The label of the job.
        :param job: Any, This is the data used to run a job.
        """
        pass

    def get(self):
        """Get a job to run.

        :returns: tuple(Label, Any)
            The Label for the job and the data needed to run the job.
        """
        pass

    def __len__(self):
        """The number of jobs waiting."""
        pass

    def remove(self, label):
        """Remove the job with this label."""
        pass


class RoundRobinScheduler(Scheduler):
    """A round robin scheduler.

    The round robin is keyed on the experiment hash so that a single experiment
    cannot hog all the resources. It doesn't prempt or anything like a scheduler
    from an OS course.
    """
    def __init__(self):
        super(RoundRobinScheduler, self).__init__()
        self.job_queue = defaultdict(deque)
        self.labels = set()
        self.rr = deque()

    def add(self, label, job):
        self.job_queue[label.exp].append((label, job))
        if label.exp in self.labels:
            return
        self.rr.append(label.exp)
        self.labels.add(label.exp)

    def get(self):
        if not self.rr:
            return None, None
        label = self.rr.popleft()
        job = self.job_queue[label].popleft()
        if not self.job_queue[label]:
            self.labels.remove(label)
            del self.job_queue[label]
            return label, job
        self.rr.append(label)
        return label, job

    def remove(self, label):
        if label.exp not in self.labels:
            return
        job_queue = self.job_queue[label.exp]
        job = None
        for (l, j) in job_queue:
            if l == label:
                job = j
        if job is None:
            return
        job_queue.remove((label, job))
        if not job_queue:
            self.labels.remove(label.exp)
            self.rr.remove(label.exp)

    def __len__(self):
        return reduce(operator.add, map(len, self.job_queue.values()))

    def __str__(self):
        s = "=" * 10
        s += "\nscheduler:\ntypes:\n\t"
        s += '\t'.join(self.rr)
        s += "\njobs:"
        for l in self.job_queue:
            s += '\n\t{}:\n\t\t'.format(l)
            s += '\t'.join(map(str, self.job_queue[l]))
        s += "\n" + "*" * 10
        return s
