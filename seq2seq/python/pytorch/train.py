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
            self.model = torch.nn.DataParallel(model).cuda()
            self.crit.cuda()
    
    def _wrap(self, src, tgt):
        dst = tgt[:,:-1]
        tgt = tgt[:,1:]
        if self.gpu:
            src = src.cuda()
            dst = dst.cuda()
            tgt = tgt.cuda()
        return Variable(src), Variable(dst), Variable(tgt)
    
    def _total(self, tgt):
        tgtt = tgt.data.long()
        return torch.sum(tgtt.ne(0))

    def test(self, ts):
        self.model.eval()
        total_loss = total = 0
        #start_time = time.time()
        steps = len(ts)
        pg = ProgressBar(steps)
        for src, tgt, src_len, tgt_len in ts:
            src, dst, tgt = self._wrap(src, tgt)
            pred = self.model((src, dst))
            loss = self.crit(pred, tgt)
            total_loss += loss.data[0]
            total += self._total(tgt)
            pg.update()
        pg.done()

        #duration = time.time() - start_time
        avg_loss = float(total_loss)/total
        #print('%s (Loss %.4f) (Perplexity %.4f) (%.3f sec)' % 
        #      (phase, avg_loss, np.exp(avg_loss), duration))
        return avg_loss

    def train(self, ts):
        self.model.train()

        #start_time = time.time()
        steps = len(ts)
        total_loss = total = 0
        pg = ProgressBar(steps)
        for src,tgt,src_len,tgt_len in ts:
            self.optimizer.zero_grad()
            src, dst, tgt = self._wrap(src, tgt)
            pred = self.model((src, dst))
            loss = self.crit(pred, tgt)
            total_loss += loss.data[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
            total += self._total(tgt)
            self.optimizer.step()
            pg.update()
        pg.done()
        #duration = time.time() - start_time

        avg_loss = float(total_loss)/total
        return avg_loss

        #print('Train (Loss %.4f) (Perplexity %.4f) (%.3f sec)' % 
        #      (avg_loss, np.exp(avg_loss), duration))

# Mashed together from code using numpy only, hacked for th Tensors
def show_examples(use_gpu, model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples):
    si = np.random.randint(0, len(es))

    src_array, tgt_array, src_len, _ = es[si]

    if max_examples > 0:
        max_examples = min(max_examples, src_array.size(0))
        src_array = src_array[0:max_examples]
        tgt_array = tgt_array[0:max_examples]
        src_len = src_len[0:max_examples]

    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']

    if use_gpu:
        src_array = src_array.cuda()
    
    for src_len,src_i,tgt_i in zip(src_len, src_array, tgt_array):

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


def fit(seq2seq, ts, es, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs
    gpu = bool(kwargs['gpu']) if 'gpu' in kwargs else True
    optim = kwargs['optim'] if 'optim' in kwargs else 'adam'
    eta = float(kwargs['eta']) if 'eta' in kwargs else 0.01
    mom = float(kwargs['mom']) if 'mom' in kwargs else 0.9
    clip = float(kwargs['clip']) if 'clip' in kwargs else 5
    model_file = kwargs['outfile'] if 'outfile' in kwargs and kwargs['outfile'] is not None else './model.pyth'
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = Trainer(gpu, seq2seq, optim, eta, mom, clip)
    val_min = 1000
    last_improved = 0

    for i in range(epochs):
        print('Training epoch %d' % (i+1))
        start_time = time.time()
        avg_train_loss = trainer.train(ts)
        duration = time.time() - start_time
        print('Training Loss %.4f (Perplexity %.4f) (%.3f sec)' % 
              (avg_train_loss, np.exp(avg_train_loss), duration))


        if after_train_fn is not None:
            after_train_fn(seq2seq)

        start_time = time.time()
        avg_val_loss = trainer.test(es)
        duration = time.time() - start_time
        print('Validation Loss %.4f (Perplexity %.4f) (%.3f sec)' % 
              (avg_val_loss, np.exp(avg_val_loss), duration))

        if avg_val_loss < val_min:
            last_improved = i
            val_min = avg_val_loss
            print('Lowest error achieved yet -- writing model')
            seq2seq.save(model_file)

        if (i - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

