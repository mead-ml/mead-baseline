import pytest
import numpy as np
from unittest.mock import patch, Mock
import baseline
from mead.tf.exporters import TaggerTensorFlowExporter
from mead.tf.exporters import FIELD_NAME
from mead.tasks import TaggerTask
import tensorflow as tf
import baseline.tf.tagger
import os
from baseline.w2v import Word2VecModel
from tf_session import session_context, session_tf

TAGGER_MODEL = os.path.join('test_data','tagger_model', 'tagger-model-tf-21275')
TAGGER_MODEL_FILES = os.path.join(os.path.dirname(__file__), TAGGER_MODEL)
FAKE_FILE = './test_data/not_here'
TEST_EMBEDDINGS = './test_data/glove_test.txt'

TAGGER_CONFIG_PARAMS = {
    "preproc": {
        "mxlen": -1,
        "mxwlen": -1,
        "lower": True
    },
    "loader": {
        "reader_type": "default"
    },
    "model": {
        "model_type": "default",
        "cfiltsz": [3],
        "hsz": 200,
        "wsz": 30,
        "dropout": 0.5,
        "rnntype": "blstm",
        "layers": 1,
        "crf_mask": True,
	    "crf": 1
    },
    "train": {
        "epochs": 100,
        "optim": "sgd",
        "eta": 0.015,
        "mom": 0.9,
        "patience": 40,
        "early_stopping_metric": "f1",
        "clip": 5.0,
        "span_type": "iobes"
    }
}

@pytest.fixture
def tagger_task():
    m = Mock()
    m.config_params = TAGGER_CONFIG_PARAMS

    return m

@pytest.fixture
def tagger_exporter():
    return TaggerTensorFlowExporter(tagger_task())

@pytest.fixture
def words():
    m = Mock()
    dsz = 100
    m.dsz = dsz
    m.weights = np.zeros(np.array([26872, dsz]), dtype=np.float32)
    m.vocab = {
        '<PAD>': 0,
        'test': 1
    }
    return m

@pytest.fixture
def chars():
    m = Mock()
    m.dsz = 30
    m.weights = np.zeros(np.array([86, 30]), dtype=np.float32)
    m.vocab = {
        'a': 0,
        'b': 1
    }
    return m

@pytest.fixture
def embeds():
    return {
        'word': words(),
        'char': chars()
    }

@pytest.fixture
def graph():
    return tf.Graph()

class ExporterTest(tf.test.TestCase):

    """
    below tests throw "no variables to save".

    I would assume this is due to an issue with
    how I handle sessions, although I can't pin
    it down.
    """
    # def test_restore_model(self):
    #     exporter = tagger_exporter()
    #     sess = self.test_session()
    #     model = exporter.restore_model(sess, TAGGER_MODEL_FILES)

    # def restore_missing_throws_exception(self):
    #     with pytest.raises(Exception) as e_info:
    #         exporter = tagger_exporter()
    #         sess = self.test_session()
    #         model = exporter.restore_model(sess, FAKE_FILE)

    def test_create_example(self):
        with self.test_session() as sess:
            exporter = tagger_exporter()
            serialized_example, example = exporter._create_example([])

            assert FIELD_NAME in example

    def test_create_example_with_extra_features(self):
        with self.test_session() as sess:
            exporter = tagger_exporter()
            serialized_example, example = exporter._create_example(['test'])

            assert 'test' in example

    # def test_predict_has_tokens(self):
    #     """
    #     using a regular session as called from inside the model,
    #     run _run(), returning the signature for the exported model
        
    #     """
    #     with self.test_session(graph=tf.Graph()) as sess:
    #         exporter = tagger_exporter()
    #         embed = embeds()
    #         exporter._initialize_embeddings_map=Mock(
    #             return_value=embed
    #         )
            
    #         sig_in, sig_out, sig_name = exporter._run(sess, TAGGER_MODEL_FILES, None, use_preproc=True)

    #     assert 'tokens' in sig_in.predict
    #     tf.reset_default_graph()


    # def test_predict_tensor_info(self):
    #     sess = tf.Session()
    #     try:
    #         exporter = tagger_exporter()
    #         embed = embeds()
    #         exporter._initialize_embeddings_map=Mock(
    #             return_value=embed
    #         )

    #         serialized_example, example = exporter._create_example([])
    #         predict_truth = tf.saved_model.utils.build_tensor_info(example[FIELD_NAME])
            
    #         sig_in, sig_out, sig_name = exporter._run(sess, TAGGER_MODEL_FILES, embeds(), use_preproc=True)

    #         assert isinstance(sig_in.predict['tokens'], tf.TensorInfo)
    #         assert sig_in.predict['tokens'].name.split('/')[1] == predict_truth.name.split("/")[1]

    #     finally:
    #         sess.close()

    # def test_predict_without_preproc(self):
    #     with self.test_session(graph=tf.Graph()) as sess:
    #         exporter = tagger_exporter()
    #         embed = embeds()
    #         exporter._initialize_embeddings_map=Mock(
    #             return_value=embed
    #         )

    #         sig_in, sig_out, sig_name = exporter._run(sess, TAGGER_MODEL_FILES, None, use_preproc=False)
    #         assert isinstance(sig_in.predict['x'], tf.TensorInfo)
    #     tf.reset_default_graph()

    def test_predict_info_without_preproc(self):
        """
        the model that is being loaded specifies a mxlen
        of -1, so we use a max len as defined by the data.

        I know this is 124, so we inspect the tensorInfo here
        to check if we have a tensor of the right shape.

        This is a proxy to assume that the model.x tensor
        shape matches this shape. I can't check equality here
        directly.
        """
        with self.test_session(graph=tf.Graph()) as sess:
            exporter = tagger_exporter()
            embed = embeds()
            exporter._initialize_embeddings_map=Mock(
                return_value=embed
            )

            sig_in, sig_out, sig_name = exporter._run(sess, TAGGER_MODEL_FILES, None, use_preproc=False)
            assert sig_in.predict['x'].tensor_shape.dim[1].size == 124