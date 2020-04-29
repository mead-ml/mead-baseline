import os
import json
import pytest

pytest.skip("There is now a crf object we should be testing on instead of the tagger", allow_module_level=True)
import numpy as np

tf = pytest.importorskip("tensorflow")
from baseline.model import create_tagger_model, load_tagger_model
from baseline.embeddings import load_embeddings
from baseline.utils import transition_mask as np_transition_mask
from baseline.tf.tfy import transition_mask


HSZ = 100
WSZ = 30
S = "<GO>"
E = "<EOS>"
P = "<PAD>"
SPAN_TYPE = "IOB2"
LOC = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="module")
def set_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    del os.environ["CUDA_VISIBLE_DEVICES"]


@pytest.fixture
def label_vocab():
    LOC = os.path.dirname(os.path.realpath(__file__))
    vocab_loc = os.path.join(LOC, "test_data", "crf_vocab")
    return json.load(open(vocab_loc))


@pytest.fixture
def embeds():
    import baseline.tf.embeddings

    embeds = {}
    embeds["word"] = load_embeddings("word", dsz=HSZ, known_vocab={chr(i): i for i in range(100)})["embeddings"]
    embeds["char"] = load_embeddings("char", dsz=WSZ, known_vocab={chr(i): i for i in range(100)})["embeddings"]
    return embeds


@pytest.fixture
def save_file():
    base = "del"
    path = os.path.join(LOC, base)
    yield path
    os.remove(os.path.join(LOC, "checkpoint"))
    for file_name in os.listdir(LOC):
        if file_name.startswith(base):
            os.remove(os.path.join(LOC, file_name))


@pytest.fixture
def model(label_vocab, embeds, mask):
    from baseline.tf import tagger

    model = create_tagger_model(
        embeds, label_vocab, crf=True, constraint_mask=mask, hsz=HSZ, cfiltsz=[3], wsz=WSZ, layers=2, rnntype="blstm"
    )
    model.create_loss()
    model.sess.run(tf.compat.v1.global_variables_initializer())
    return model


@pytest.fixture
def mask(label_vocab):
    return transition_mask(label_vocab, SPAN_TYPE, label_vocab[S], label_vocab[E], label_vocab[P])


def test_mask_used(label_vocab, model):
    transition = model.sess.run(model.A)
    assert transition[label_vocab["O"], label_vocab[S]] == -1e4


def test_mask_is_transpose(label_vocab, model):
    transition = model.sess.run(model.mask)
    np_mask = np_transition_mask(label_vocab, SPAN_TYPE, label_vocab[S], label_vocab[E], label_vocab[P])
    np.testing.assert_allclose(transition.T, np_mask)


def test_persists_save(model, save_file):
    model.save_using(tf.compat.v1.train.Saver())
    t1 = model.sess.run(model.A)
    model.save(save_file)
    m2 = load_tagger_model(save_file)
    t2 = model.sess.run(m2.A)
    np.testing.assert_allclose(t1, t2)


def test_skip_mask(label_vocab, embeds, mask):
    from baseline.tf import tagger

    model = create_tagger_model(embeds, label_vocab, crf=True, hsz=HSZ, cfiltsz=[3], wsz=WSZ, layers=2, rnntype="blstm")
    model.create_loss()
    model.sess.run(tf.compat.v1.global_variables_initializer())
    transition = model.sess.run(model.A)
    assert transition[label_vocab["O"], label_vocab[S]] != -1e4
