import operator
from functools import reduce
from collections import defaultdict, deque


class Scheduler(object):
    def __init__(self):
        super(Scheduler, self).__init__()

    def add(self, label, job):
        pass

    def get(self):
        pass

    def __len__(self):
        pass


class RoundRobinScheduler(Scheduler):
    def __init__(self):
        super(RoundRobinScheduler, self).__init__()
        self.job_queue = defaultdict(deque)
        self.labels = set()
        self.rr = deque()

    def add(self, label, job):
        self.job_queue[label].append(job)
        if label in self.labels:
            return
        self.rr.append(label)
        self.labels.add(label)

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
