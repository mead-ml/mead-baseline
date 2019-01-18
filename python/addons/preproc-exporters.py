import os
from mead.exporters import register_exporter
from baseline.tf.embeddings import *
from preprocessors import Token1DPreprocessorCreator
from baseline.utils import export
from collections import namedtuple
from mead.tf.exporters import ClassifyTensorFlowExporter, create_assets

__all__ = []
exporter = export(__all__)

SignatureOutput = namedtuple("SignatureOutput", ("classes", "scores"))


@exporter
@register_exporter(task='classify', name='preproc')
class ClassifyTensorFlowPreProcExporter(ClassifyTensorFlowExporter):

    def __init__(self, task):
        super(ClassifyTensorFlowPreProcExporter, self).__init__(task)

    def _create_rpc_call(self, sess, model_file):
        model_base_dir = os.path.split(model_file)[0]
        pid = model_file.split("-")[-1]
        features = ["word"]
        preprocessor = Token1DPreprocessorCreator(model_base_dir, pid, features)
        model_params = self.task.config_params["model"]
        tf_example, preprocessed = preprocessor.run()
        for feature in preprocessed:
            model_params[feature] = preprocessed[feature]
        model, classes, values = self._create_model(sess, model_file)
        sig_input = {'tokens': tf.saved_model.utils.build_tensor_info(tf_example[preprocessor.FIELD_NAME])}
        sig_output = SignatureOutput(classes, values)
        sig_name = 'predict_text'
        assets = create_assets(model_file, sig_input, sig_output, sig_name, model.lengths_key)
        return sig_input, sig_output, sig_name, assets

