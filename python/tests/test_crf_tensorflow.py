import os
import json
import pytest
import numpy as np
tf = pytest.importorskip('tensorflow')
from baseline.w2v import RandomInitVecModel
from baseline.model import create_tagger_model
from baseline.utils import crf_mask as np_crf
from baseline.tf.tagger import RNNTaggerModel

os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
def embeds():
    embeds = {}
    embeds['word'] = RandomInitVecModel(HSZ, {chr(i): i for i in range(100)})
    embeds['char'] = RandomInitVecModel(WSZ, {chr(i): i for i in range(100)})
    return embeds

@pytest.fixture
def save_file():
    base = 'del'
    path = os.path.join(LOC, base)
    yield path
    os.remove(path + "-char.vocab")
    os.remove(path + "-word.vocab")
    os.remove(os.path.join(LOC, 'checkpoint'))
    for file_name in os.listdir(LOC):
        if file_name.startswith(base):
            os.remove(os.path.join(LOC, file_name))

@pytest.fixture
def model(label_vocab, embeds):
    tf.reset_default_graph()
    model = create_tagger_model(
        label_vocab, embeds,
        crf=True, crf_mask=True, span_type=SPAN_TYPE,
        hsz=HSZ, cfiltsz=[3], wsz=WSZ,
        layers=2, rnntype="blstm"
    )
    model.create_loss()
    model.sess.run(tf.global_variables_initializer())
    return model

def test_mask_used(label_vocab, model):
    transition = model.sess.run(model.A)
    assert transition[label_vocab['O'], label_vocab[S]] == -1e4

def test_mask_is_transpose(label_vocab, model):
    transition = model.sess.run(model.mask)
    np_mask = np_crf(label_vocab, SPAN_TYPE, label_vocab[S], label_vocab[E], label_vocab[P])
    np.testing.assert_allclose(transition.T, np_mask)

def test_persists_save(model, save_file):
    model.save_using(tf.train.Saver())
    t1 = model.sess.run(model.A)
    model.save(save_file)
    m2 = RNNTaggerModel.load(save_file)
    t2 = model.sess.run(m2.A)
    np.testing.assert_allclose(t1, t2)

def test_skip_mask(label_vocab, embeds):
    tf.reset_default_graph()
    model = create_tagger_model(
        label_vocab, embeds,
        crf=True, span_type=SPAN_TYPE,
        hsz=HSZ, cfiltsz=[3], wsz=WSZ,
        layers=2, rnntype="blstm"
    )
    model.create_loss()
    model.sess.run(tf.global_variables_initializer())
    transition = model.sess.run(model.A)
    assert transition[label_vocab['O'], label_vocab[S]] != -1e4

def test_error_on_mask_and_no_span(label_vocab, embeds):
    tf.reset_default_graph()
    model = create_tagger_model(
        label_vocab, embeds,
        crf=True, crf_mask=True,
        hsz=HSZ, cfiltsz=[3], wsz=WSZ,
        layers=2, rnntype="blstm"
    )
    with pytest.raises(AssertionError):
        model.create_loss()
