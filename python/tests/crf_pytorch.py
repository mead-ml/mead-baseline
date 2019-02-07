import os
import json
import pytest
import numpy as np
torch = pytest.importorskip('torch')
from torch.optim import SGD


HSZ = 100
WSZ = 30
S = '<GO>'
E = '<EOS>'
P = '<PAD>'
SPAN_TYPE="IOB2"
LOC = os.path.dirname(os.path.realpath(__file__))


# @pytest.fixture
# def label_vocab():
#     LOC = os.path.dirname(os.path.realpath(__file__))
#     vocab_loc = os.path.join(LOC, "test_data", "crf_vocab")
#     return json.load(open(vocab_loc))


# @pytest.fixture
# def crf(label_vocab):
#     from baseline.pytorch.crf import CRF, transition_mask
#     mask = transition_mask(
#         label_vocab, SPAN_TYPE,
#         label_vocab[S], label_vocab[E], label_vocab[P]
#     )
#     return CRF(
#         len(label_vocab),
#         (label_vocab[S], label_vocab[E]), True,
#         mask
#     )


# def test_mask_is_applied(label_vocab, crf):
#     t = crf.transitions.detach().numpy()
#     assert t[0, label_vocab['<GO>'], label_vocab['O']] == -1e4


# def test_mask_skipped(label_vocab):
#     from baseline.pytorch.crf import CRF
#     crf = CRF(
#         len(label_vocab),
#         (label_vocab[S], label_vocab[E]),
#     )
#     t = crf.transitions.detach().numpy()
#     assert t[0, label_vocab['<GO>'], label_vocab['O']] != -1e4


# def test_mask_same_after_update(label_vocab, crf):
#     crf = crf.cuda()
#     opt = SGD(crf.parameters(), lr=0.01)
#     m1 = crf.mask.cpu().numpy()
#     t1 = crf.transitions_p.cpu().detach().numpy()
#     gold = torch.LongTensor([[3, 9, 9, 4, 6, 7, 5]]).cuda()
#     emissions = torch.rand(1, gold.shape[1], len(label_vocab)).cuda()
#     l = crf.neg_log_loss(emissions, gold, torch.LongTensor([gold.shape[1]]).cuda())
#     l.backward()
#     opt.step()
#     m2 = crf.mask.cpu().numpy()
#     t2 = crf.transitions_p.cpu().detach().numpy()
#     np.testing.assert_allclose(m1, m2)
#     with pytest.raises(AssertionError):
#         np.testing.assert_allclose(t1, t2)
