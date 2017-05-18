import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import math
import numpy as np
import data
from utils import ProgressBar, lookup_sentence
from torchy import long_tensor_alloc, tensor_shape, tensor_max

class Trainer:

    def __init__(self, gpu, model, optim, eta, mom, clip):
        self.gpu = gpu
        self.clip = clip
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
        dst = ds["tgt"][:,:-1]
        tgt = ds["tgt"][:,1:]

        if self.gpu:
            src = src.cuda()
            dst = dst.cuda()
            tgt = tgt.cuda()
        return Variable(src), Variable(dst), Variable(tgt)
    
    def _total(self, tgt):
        tgtt = tgt.data.long()
        return torch.sum(tgtt.ne(0))

    def test(self, ts, batchsz, phase='Test'):

        self.model.eval()

        total_loss = total = 0
        start_time = time.time()
        steps = int(math.floor(len(ts)/float(batchsz)))

        for i in range(steps):
            ts_i = data.batch(ts, i, batchsz, long_tensor_alloc, tensor_shape, tensor_max)
            src, dst, tgt = self._wrap(ts_i)
            pred = self.model((src, dst))
            loss = self.crit(pred, tgt)
            total_loss += loss.data[0]
            total += self._total(tgt)

        duration = time.time() - start_time
        avg_loss = float(total_loss)/total
        print('%s (Loss %.4f) (Perplexity %.4f) (%.3f sec)' % 
              (phase, avg_loss, np.exp(avg_loss), duration))
        return avg_loss

    def train(self, ts, batchsz):
        self.model.train()

        start_time = time.time()

        steps = int(math.floor(len(ts)/float(batchsz)))
        shuffle = np.random.permutation(np.arange(steps))
        total_loss = total = 0
        pg = ProgressBar(steps)
        for i in range(steps):
            self.optimizer.zero_grad()

            si = shuffle[i]
            ts_i = data.batch(ts, si, batchsz, long_tensor_alloc, tensor_shape, tensor_max)
            src, dst, tgt = self._wrap(ts_i)
            pred = self.model((src, dst))
            loss = self.crit(pred, tgt)
            total_loss += loss.data[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)

            total += self._total(tgt)
            self.optimizer.step()
            pg.update()
        pg.done()
        duration = time.time() - start_time

        avg_loss = float(total_loss)/total

        print('Train (Loss %.4f) (Perplexity %.4f) (%.3f sec)' % 
              (avg_loss, np.exp(avg_loss), duration))

# Mashed together from code using numpy only, hacked for th Tensors
def show_examples(use_gpu, model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples):

    batch = data.batch(es, 0, max_examples, long_tensor_alloc, tensor_shape, tensor_max)
    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']

    src_array = batch['src']
    tgt_array = batch['tgt']
    if use_gpu:
        src_array = src_array.cuda()
    
    for src_i,tgt_i in zip(src_array, tgt_array):

        print('========================================================================')
        sent = lookup_sentence(rlut1, src_i.cpu().numpy(), reverse=True)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        dst_i = torch.zeros(1, mxlen).long()
        if use_gpu:
            dst_i = dst_i.cuda()

        next_value = GO
        src_i = src_i.view(1, -1)
        for j in range(mxlen):
            dst_i[0,j] = next_value
            probv = model((Variable(src_i), Variable(dst_i)))
            output = probv.squeeze()[j]
            if sample is False:
                _, next_value = torch.max(output, 0)
                next_value = int(next_value.data[0])
            else:
                probs = output.data.exp()
                # This is going to zero out low prob. events so they are not
                # sampled from
                best, ids = probs.topk(prob_clip, 0, largest=True, sorted=True)
                probs.zero_()
                probs.index_copy_(0, ids, best)
                probs.div_(torch.sum(probs))
                fv = torch.multinomial(probs, 1)[0]
                next_value = fv

            if next_value == EOS:
                break

        sent = lookup_sentence(rlut2, dst_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')
