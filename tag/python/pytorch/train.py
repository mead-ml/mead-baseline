import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import math
import numpy as np
import data
from torchy import long_tensor_alloc, tensor_shape

class Trainer:

    def __init__(self, gpu, model, optim, eta, mom):
        self.gpu = gpu
        if optim == 'adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(), lr=eta)
        elif optim == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=eta)
        elif optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=eta)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=mom)
        self.model = model

        self.crit = model.create_loss()
        if gpu:
            self.model.cuda()
            self.crit.cuda()
    
    def _batch(self, ts, si, batchsz):
        ds = data.batch(ts, si, batchsz, long_tensor_alloc, tensor_shape)
        xch = ds["xch"]
        x = ds["x"]
        y = ds["y"]

        if self.gpu:
            xch = xch.cuda()
            x = x.cuda()
            y = y.cuda()
        return Variable(xch), Variable(x), Variable(y)
    
    # Ok, accuracy only for now, B x T x C
    def _right(self, pred, y):
        _, best = pred.max(2)
        best = best.data.long().squeeze()
        yt = y.data.long()
        mask = yt.ne(0)
        return torch.sum(mask * (yt == best))

    def _total(self, y):
        yt = y.data.long()
        return torch.sum(yt.ne(0))

    def test(self, ts, batchsz=1, phase='Test'):

        self.model.eval()

        total_loss = total_corr = total = 0
        start_time = time.time()
        steps = int(math.floor(len(ts)/float(batchsz)))

        for i in range(steps):
            xch, x, y = self._batch(ts, i, batchsz)
            pred = self.model((xch, x))
            loss = self.crit(pred, y)
            total_loss += loss.data[0]

            total_corr += self._right(pred, y)
            total += self._total(y)

        duration = time.time() - start_time

        test_acc = float(total_corr)/total

        print('%s (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % 
              (phase, float(total_loss)/total, total_corr, total, test_acc, duration))
        return test_acc

    def train(self, ts, batchsz=1):
        self.model.train()

        start_time = time.time()

        steps = int(math.floor(len(ts)/float(batchsz)))

        shuffle = np.random.permutation(np.arange(steps))

        total_loss = total_corr = total = 0
        for i in range(steps):
            self.optimizer.zero_grad()
            si = shuffle[i]
            xch, x, y = self._batch(ts, si, batchsz)
            pred = self.model((xch, x))
            loss = self.crit(pred, y)
            total_loss += loss.data[0]
            loss.backward()

            total_corr += self._right(pred, y)
            total += self._total(y)
            self.optimizer.step()

        duration = time.time() - start_time

        print('Train (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % 
              (float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))
