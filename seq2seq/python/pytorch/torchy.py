import torch
import numpy as np
from utils import lookup_sentence
from torch.autograd import Variable

def tensor_shape(tensor):
    return tensor.size()

def long_0_tensor_alloc(dims):
    lt = long_tensor_alloc(dims)
    lt.zero_()
    return lt

def long_tensor_alloc(dims):
    if type(dims) == int or len(dims) == 1:
        return torch.LongTensor(dims)
    return torch.LongTensor(*dims)

# Mashed together from code using numpy only, hacked for th Tensors
def show_batch(use_gpu, model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples):
    sz = len(es)
    rnum = int((sz - 1) * np.random.random_sample())
    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']
    
    batch = es[rnum]

    src_array = batch['src']
    tgt_array = batch['tgt']
    if use_gpu:
        src_array = src_array.cuda()
    
    i = 0
    for src_i,tgt_i in zip(src_array, tgt_array):

        if i > max_examples:
            break
        i += 1
        print('========================================================================')

        sent = lookup_sentence(rlut1, src_i)
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
            probv = model.forward((Variable(src_i), Variable(dst_i)))
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
