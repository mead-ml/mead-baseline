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
from utils import ProgressBar

class Trainer:

    def __init__(self, gpu, model, optim, eta, mom):
        self.gpu = gpu
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

    def _add_to_cm(self, cm, y, pred):
        _, best = pred.max(1)
        yt = y.cpu().int()
        yp = best.cpu().int().squeeze()
        cm.add_batch(yt.data.numpy(), yp.data.numpy())

    def test(self, ts, cm, batchsz=1, phase='Test'):
        self.model.eval()

        total_loss = 0
        start_time = time.time()
        steps = int(math.floor(len(ts)/float(batchsz)))
        cm.reset()

        for i in range(steps):
            x, y = self._batch(ts, i, batchsz)
            pred = self.model(x)
            loss = self.crit(pred, y)
            total_loss += loss.data[0]
            self._add_to_cm(cm, y, pred)
       
        duration = time.time() - start_time

        total_corr = cm.get_correct()
        total = cm.get_total()
        test_acc = float(total_corr)/total
        print('%s (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % 
              (phase, float(total_loss)/total, total_corr, total, test_acc, duration))
        print(cm)
        return test_acc

    def train(self, ts, cm, batchsz=1):
        self.model.train()

        start_time = time.time()

        steps = int(math.floor(len(ts)/float(batchsz)))

        shuffle = np.random.permutation(np.arange(steps))
        pg = ProgressBar(steps)
        cm.reset()

        total_loss = 0
        for i in range(steps):
            self.optimizer.zero_grad()
            si = shuffle[i]
            x, y = self._batch(ts, si, batchsz)
            pred = self.model(x)
            loss = self.crit(pred, y)
            total_loss += loss.data[0]
            loss.backward()
            self._add_to_cm(cm, y, pred)
            self.optimizer.step()
            pg.update()
        pg.done()

        duration = time.time() - start_time
        total_corr = cm.get_correct()
        total = cm.get_total()

        print('Train (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % 
              (float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))
        print(cm)

