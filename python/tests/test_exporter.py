import os
import pytest
import numpy as np
from mock import patch, Mock
tf = pytest.importorskip('tensorflow')
from mead.tasks import TaggerTask

CONFIG_PARAMS = {
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
def task():
    m = Mock()
    m.config_params = CONFIG_PARAMS
    return m

@pytest.fixture
def tagger_exporter():
    from mead.tf.exporters import TaggerTensorFlowExporter
    m = TaggerTensorFlowExporter(task())
    return m

@pytest.fixture
def s2s_exporter():
    from mead.tf.exporters import Seq2SeqTensorFlowExporter
    m = Seq2SeqTensorFlowExporter(task())
    return m

@pytest.fixture
def cls_exporter():
    from mead.tf.exporters import ClassifyTensorFlowExporter
    m = ClassifyTensorFlowExporter(task())
    return m

def fake_read_json(basename):
    return {'vsz': 1, 'dsz': 1}


exporter_sequence = [s2s_exporter, tagger_exporter, cls_exporter]

class ExporterTest(tf.test.TestCase):

    @patch('mead.tf.exporters.read_json', side_effect=fake_read_json)
    @patch('mead.tf.exporters.eval', side_effect=Mock)
    def test_init_embeddings(self, read_json, eval):
        with self.test_session() as sess:
            exporter = s2s_exporter()

            out = exporter.init_embeddings([('test', 'class'), ('test2', 'class')], 'basename')

            assert 'test' in out and 'test2' in out


    @patch('mead.tf.exporters.tf.saved_model.utils.build_tensor_info', return_value='test_info')
    def test_create_rpc_call_exception(self, build_tensor_info):
        """
        I segfault unless I have this patch applied, but I don't think it does anything.
        """
        for e in exporter_sequence:
            with self.subTest():
                exporter = e()

                def fake_create_model_exception(sess, basename):
                    model = Mock()
                    model.src_embeddings = {'one': 1, 'two': 2}
                    model.embeddings = {'one': 1, 'two': 2}

                    return model, Mock(), Mock()

                exporter._create_model = Mock(side_effect=fake_create_model_exception)
                sess = Mock()

                with pytest.raises(Exception):
                    sin, sout, sig = exporter._create_rpc_call(sess, 'basename')

    @patch('mead.tf.exporters.tf.saved_model.utils.build_tensor_info', return_value='test_info')
    def test_create_rpc_call(self, build_tensor_info):
        """
        I segfault unless I have this patch applied, but I don't think it does anything.
        """
        for e in exporter_sequence:
            with self.subTest():
                exporter = e()

                mclasses = Mock()
                mvalues = Mock()
                def fake_create_model_exception(sess, basename):
                    model = Mock()

                    info = Mock()
                    info.x = 1
                    model.src_embeddings = {'one': info, 'two': info}
                    model.embeddings = {'one': info, 'two': info}

                    return model, mclasses, mvalues

                exporter._create_model = Mock(side_effect=fake_create_model_exception)
                sess = Mock()

                sin, sout, sig = exporter._create_rpc_call(sess, 'basename')

                assert 'one' in sin and 'two' in sin
                assert sout.classes == mclasses

    # def test_create_rpc_call(self):
    #     with self.test_session() as sess:
    #         exporter = s2s_exporter()

    #         def fake_create_model(sess, basename):
    #             model = Mock()
    #             model.src_embeddings = {'one': 'one_out', 'two': 'two_out'}

    #             return model, Mock(), Mock()

    #         exporter._create_model = Mock(side_effect=fake_create_model)
    #         sess = Mock()

    #         with patch('mead.tf.exporters.tf.saved_model.utils', autospec=True) as c:
    #             c.build_tensor_info = Mock(return_value='test_info')
    #             sin, sout, sig = exporter._create_rpc_call(sess, 'basename')

    # def test_create_example(self):
    #     from mead.tf.exporters import FIELD_NAME
    #     with self.test_session() as sess:
    #         exporter = tagger_exporter()
    #         serialized_example, example = exporter._create_example([])

    #         assert FIELD_NAME in example

    # def test_create_example_with_extra_features(self):
    #     with self.test_session() as sess:
    #         exporter = tagger_exporter()
    #         serialized_example, example = exporter._create_example(['test'])

    #         assert 'test' in example

    # def test_predict_has_tokens(self):
    #     """
    #     using a regular session as called from inside the model,
    #     run _run(), returning the signature for the exported model

    #     we check if the preprocessing fields returned in our mocked
    #     _run_preproc() method are used to call the signature.
    #     """
    #     with self.test_session() as sess:
    #         exporter = tagger_exporter()
    #         mocked_model = Mock()
    #         exporter._create_model = Mock(return_value=(None, None, mocked_model))
    #         exporter.restore_checkpoint = Mock()
    #         exporter._run_preproc = Mock(return_value=('srl', 'ex', 'raw', 'lengths'))


    #         with patch('mead.tf.exporters.SignatureInput') as sigin:
    #             sig_in, sig_out, sig_name = exporter._run(sess, TAGGER_MODEL_FILES, None, use_preproc=True)

    #             sigin.assert_called_once_with('srl', 'raw', [])

    # def test_predict_info_without_preproc(self):
    #     """
    #     test that we don't use preprocessing, and that our
    #     signature input object is called assuming as much.
    #     """
    #     with self.test_session() as sess:
    #         exporter = tagger_exporter()
    #         mocked_model = Mock()
    #         exporter._create_model = Mock(return_value=(None, None, mocked_model))
    #         exporter.restore_checkpoint = Mock()
    #         exporter._run_preproc = Mock(return_value=('srl', 'ex', 'raw', 'lengths'))


    #         with patch('mead.tf.exporters.SignatureInput') as sigin:
    #             sig_in, sig_out, sig_name = exporter._run(sess, TAGGER_MODEL_FILES, None, use_preproc=False)

    #             assert not exporter._run_preproc.called
    #             sigin.assert_called_once_with(None, None, ['lengths'], model=mocked_model)
