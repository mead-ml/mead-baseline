import os
import json
import pytest
import numpy as np
torch = pytest.importorskip('torch')
from torch.optim import SGD
from baseline.w2v import RandomInitVecModel
from baseline.pytorch.tagger.model import create_model
from baseline.pytorch.torchy import CRF, crf_mask
from baseline.utils import crf_mask as np_crf

HSZ = 100
WSZ = 30
S = '<GO>'
E = '<EOS>'
P = '<PAD>'
SPAN_TYPE="IOB2"
LOC = os.path.dirname(os.path.realpath(__file__))

@pytest.fixture
def label_vocab():
    LOC = os.path.dirname(os.path.realpath(__file__))
    vocab_loc = os.path.join(LOC, "test_data", "crf_vocab")
    return json.load(open(vocab_loc))

@pytest.fixture
def crf(label_vocab):
    return CRF(
        len(label_vocab),
        (label_vocab[S], label_vocab[E]),
        label_vocab, SPAN_TYPE, label_vocab[P]
    )

@pytest.fixture
def embeds():
    embeds = {}
    embeds['word'] = RandomInitVecModel(HSZ, {chr(i): i for i in range(100)})
    embeds['char'] = RandomInitVecModel(WSZ, {chr(i): i for i in range(100)})
    return embeds

@pytest.fixture
def model(label_vocab, embeds):
    return create_model(
        label_vocab, embeds,
        crf=True, crf_mask=True, span_type=SPAN_TYPE,
        hsz=HSZ, cfiltsz=[3], wsz=WSZ,
        layers=2, rnntype="blstm"
    )

def test_mask_is_applied(label_vocab, crf):
    t = crf.transitions.detach().numpy()
    assert t[label_vocab['<GO>'], label_vocab['O']] == -1e4

def test_mask_skipped(label_vocab):
    crf = CRF(
        len(label_vocab),
        (label_vocab[S], label_vocab[E]),
    )
    t = crf.transitions.detach().numpy()
    assert t[label_vocab['<GO>'], label_vocab['O']] != -1e4

def test_error_without_type(label_vocab):
    with pytest.raises(AssertionError):
        _ = CRF(
            len(label_vocab),
            (label_vocab[S], label_vocab[E]),
            label_vocab
        )

# Using .cuda() in pytest call is having problems
# From turning CUDA_VISIBLE_DEVICES off for tensorflow?

# def test_mask_follows_crf_device(crf):
#     assert crf.mask.device == crf.transitions_p.device
#     crf = crf.cuda()
#     assert crf.mask.device == crf.transitions_p.device

# def test_mask_same_after_update(label_vocab, crf):
#     crf = crf.cuda()
#     opt = SGD(crf.parameters(), lr=0.01)
#     m1 = crf.mask.cpu().numpy()
#     t1 = crf.transitions_p.cpu().detach().numpy()
#     gold = torch.LongTensor([3, 9, 9, 4, 6, 7, 5]).cuda()
#     emissions = torch.rand(len(gold), len(label_vocab)).cuda()
#     l = crf.neg_log_loss(emissions, gold)
#     l.backward()
#     opt.step()
#     m2 = crf.mask.cpu().numpy()
#     t2 = crf.transitions_p.cpu().detach().numpy()
#     np.testing.assert_allclose(m1, m2)
#     with pytest.raises(AssertionError):
#         np.testing.assert_allclose(t1, t2)

def test_mask_used_in_model(label_vocab, model):
    t = model.crf.transitions.detach().numpy()
    assert t[label_vocab['<GO>'], label_vocab['O']] == -1e4

def test_mask_not_used_in_model(label_vocab, embeds):
    model = create_model(
        label_vocab, embeds,
        crf=True,
        hsz=HSZ, cfiltsz=[3], wsz=WSZ,
        layers=2, rnntype="blstm"
    )
    t = model.crf.transitions.detach().numpy()
    assert t[label_vocab['<GO>'], label_vocab['O']] != -1e4

def test_error_when_mask_and_no_span(label_vocab, embeds):
    with pytest.raises(AssertionError):
        model = create_model(
            label_vocab, embeds,
            crf=True, crf_mask=True,
            hsz=HSZ, cfiltsz=[3], wsz=WSZ,
            layers=2, rnntype="blstm"
        )
