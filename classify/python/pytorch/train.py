import time
import math
import numpy as np
import torch
import torch.optim
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import data
from torch.autograd import Variable
from torchy import long_tensor_alloc, TorchExamples

class Trainer:

    def __init__(self, gpu, model, optim, eta, mom):
        self.gpu = gpu
        #parameters = model.parameters()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if optim == 'adadelta':
            print('Using adadelta, ignoring learning rate')
            self.optimizer = torch.optim.Adadelta(parameters)
        elif optim == 'adam':
            self.optimizer = torch.optim.Adam(parameters)
        else:
            self.optimizer = torch.optim.SGD(parameters, lr=eta, momentum=mom)
        self.model = model
        self.crit = model.create_loss()
        if gpu:
            self.model.cuda()
            self.crit.cuda()
    
    def _batch(self, ts, si, batchsz):
        ds = data.batch(ts, si, batchsz, vec_alloc=long_tensor_alloc, ExType=TorchExamples)
        x = ds.x
        y = ds.y
        if self.gpu:
            x = x.cuda()
            y = y.cuda()
        return Variable(x), Variable(y)

    def _right(self, pred, y):
        _, best = pred.max(1)
        best = best.data.long().squeeze()
        return torch.sum(y.data.long() == best)

    def test(self, ts, batchsz=1, phase='Test'):
        self.model.eval()

        total_loss = total_corr = total = 0
        start_time = time.time()
        steps = int(math.floor(len(ts)/float(batchsz)))

        for i in range(steps):
            x, y = self._batch(ts, i, batchsz)
            pred = self.model(x)
            loss = self.crit(pred, y)
            total_loss += loss.data[0]

            total_corr += self._right(pred, y)
            total += batchsz

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
            x, y = self._batch(ts, si, batchsz)
            pred = self.model(x)
            loss = self.crit(pred, y)
            total_loss += loss.data[0]
            loss.backward()

            total_corr += self._right(pred, y)
            total += batchsz
            self.optimizer.step()

        duration = time.time() - start_time

        print('Train (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % 
              (float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))

