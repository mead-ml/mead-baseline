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

TAGGER_MODEL = os.path.join('test_data','tagger_model', 'tagger-model-tf-21855')
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
    m = TaggerTensorFlowExporter(tagger_task())
    return m


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

    def test_predict_has_tokens(self):
        """
        using a regular session as called from inside the model,
        run _run(), returning the signature for the exported model
        
        we check if the preprocessing fields returned in our mocked
        _run_preproc() method are used to call the signature.
        """
        with self.test_session() as sess:
            exporter = tagger_exporter()
            mocked_model = Mock()
            exporter._create_model = Mock(return_value=(None, None, mocked_model))
            exporter.restore_model = Mock()
            exporter._run_preproc = Mock(return_value=('srl', 'ex', 'raw', 'lengths'))

            
            with patch('mead.tf.exporters.SignatureInput') as sigin:
                sig_in, sig_out, sig_name = exporter._run(sess, TAGGER_MODEL_FILES, None, use_preproc=True)

                sigin.assert_called_once_with('srl', 'raw', [])

    def test_predict_info_without_preproc(self):
        """
        test that we don't use preprocessing, and that our
        signature input object is called assuming as much.
        """
        with self.test_session() as sess:
            exporter = tagger_exporter()
            mocked_model = Mock()
            exporter._create_model = Mock(return_value=(None, None, mocked_model))
            exporter.restore_model = Mock()
            exporter._run_preproc = Mock(return_value=('srl', 'ex', 'raw', 'lengths'))


            with patch('mead.tf.exporters.SignatureInput') as sigin:
                sig_in, sig_out, sig_name = exporter._run(sess, TAGGER_MODEL_FILES, None, use_preproc=False)

                assert not exporter._run_preproc.called
                sigin.assert_called_once_with(None, None, ['lengths'], model=mocked_model)
