import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import math
import numpy as np
import data

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
    
    def _wrap(self, ds):
        src = ds["src"]
        dst = ds["dst"]
        tgt = ds["tgt"]

        if self.gpu:
            src = src.cuda()
            dst = dst.cuda()
            tgt = tgt.cuda()
        return Variable(src), Variable(dst), Variable(tgt)
    
    # Ok, accuracy only for now, B x T x C
    def _right(self, pred, tgt):
        _, best = pred.max(2)
        best = best.data.long().squeeze()
        tgtt = tgt.data.long()
        mask = tgtt.ne(0)
        return torch.sum(mask * (tgtt == best))

    def _total(self, tgt):
        tgtt = tgt.data.long()
        return torch.sum(tgtt.ne(0))

    def test(self, ts, phase='Test'):

        self.model.eval()

        total_loss = total_corr = total = 0
        start_time = time.time()
        steps = len(ts)

        for i in range(steps):
            src, dst, tgt = self._wrap(ts[i])
            pred = self.model((src, dst))
            loss = self.crit(pred, tgt)
            total_loss += loss.data[0]
            total_corr += self._right(pred, tgt)
            total += self._total(tgt)

        duration = time.time() - start_time
        test_acc = float(total_corr)/total

        print('%s (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % 
              (phase, float(total_loss)/total, total_corr, total, test_acc, duration))
        return test_acc

    def train(self, ts):
        self.model.train()

        start_time = time.time()

        steps = int(len(ts))
        shuffle = np.random.permutation(np.arange(steps))

        total_loss = total_corr = total = 0
        for i in range(steps):
            self.optimizer.zero_grad()
            si = shuffle[i]
            src, dst, tgt = self._wrap(ts[si])
            pred = self.model((src, dst))
            loss = self.crit(pred, tgt)
            total_loss += loss.data[0]
            loss.backward()

            total_corr += self._right(pred, tgt)
            total += self._total(tgt)
            self.optimizer.step()

        duration = time.time() - start_time

        print('Train (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % 
              (float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))
