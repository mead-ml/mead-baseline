import os
from mead.exporters import register_exporter
from mead.preprocessors import BASELINE_PREPROCESSORS, create_preprocessors
from baseline.tf.embeddings import *
#from preprocessors import Token1DPreprocessor, Token2DPreprocessor
from baseline.utils import export
from collections import namedtuple
from mead.tf.exporters import ClassifyTensorFlowExporter, TaggerTensorFlowExporter, create_assets
from collections import OrderedDict

__all__ = []
exporter = export(__all__)

SignatureOutput = namedtuple("SignatureOutput", ("classes", "scores"))


class PreProcessorController(object):
    def __init__(self, model_base_dir, pid, features):
        saved_vectorizers = self.get_vectorizers(model_base_dir, pid)
        feature_vectorizer_mapping = {feature['name']: feature['vectorizer']['type'] for feature in features}
        self.preprocessors = dict()
        for feature in feature_vectorizer_mapping:
            self.preprocessors[feature] = create_preprocessors(preprocessor_type=feature_vectorizer_mapping[feature],
                                                              model_base_dir=model_base_dir, pid=pid, feature=feature,
                                                              vectorizer=saved_vectorizers[feature])
        self.FIELD_NAME = 'tokens'

    @staticmethod
    def get_vectorizers(model_base_dir, pid):
        """
        :model_file the path-like object to the model and model name.
        :vocab_suffixes the list of vocab types. e.g. 'word', 'char', 'ner'.
        """
        import pickle
        return pickle.load(open(os.path.join(model_base_dir, "vectorizers-{}.pkl".format(pid)), "rb"))

    def _create_example(self):
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            self.FIELD_NAME: tf.FixedLenFeature(shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        return tf_example

    def preproc(self, tf_example):
        preprocessed_inputs = {}
        for feature in self.preprocessors:
            preprocessed_inputs[feature] = self.preprocessors[feature].preproc(tf_example)
        return {feature: self.preprocessors[feature].run for feature in self.preprocessors}

    def create_preprocessed_input(self, tf_example):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        types = {f: tf.int64 for f in self.preprocessors}
        return tf.map_fn(
            self.preproc, tf_example,
            dtype=types, back_prop=False
        )

    def run(self):
        tf_example = self._create_example()
        preprocessed = self.create_preprocessed_input(tf_example)
        return tf_example, preprocessed


@exporter
@register_exporter(task='classify', name='preproc')
class ClassifyTensorFlowPreProcExporter(ClassifyTensorFlowExporter):

    def __init__(self, task):
        super(ClassifyTensorFlowPreProcExporter, self).__init__(task)

    def _create_rpc_call(self, sess, model_file):
        model_base_dir = os.path.split(model_file)[0]
        pid = model_file.split("-")[-1]
        pc = PreProcessorController(model_base_dir, pid, self.task.config_params['features'])
        model_params = self.task.config_params['model']
        tf_example, preprocessed = pc.run()
        print(preprocessed)
        sys
        for feature in preprocessed:
            model_params[feature] = preprocessed[feature]
        model, classes, values = self._create_model(sess, model_file)
        sig_input = {'tokens': tf.saved_model.utils.build_tensor_info(tf_example[pc.FIELD_NAME])}
        sig_output = SignatureOutput(classes, values)
        sig_name = 'predict_text'
        assets = create_assets(model_file, sig_input, sig_output, sig_name, model.lengths_key)
        return sig_input, sig_output, sig_name, assets


@exporter
@register_exporter(task='tagger', name='preproc')
class TaggerTensorFlowPreProcExporter(TaggerTensorFlowExporter):

    def __init__(self, task):
        super(TaggerTensorFlowPreProcExporter, self).__init__(task)

    def _create_rpc_call(self, sess, model_file):
        model_base_dir = os.path.split(model_file)[0]
        pid = model_file.split("-")[-1]
        features = [x['name'] for x in self.task.config_params['features']]
        preprocessor = Token2DPreprocessor(model_base_dir, pid, features)
        model_params = self.task.config_params["model"]
        tf_example, preprocessed = preprocessor.run()
        for feature in preprocessed:
            model_params[feature] = preprocessed[feature]
        model, classes, values = self._create_model(sess, model_file)
        sig_input = {
            'tokens': tf.saved_model.utils.build_tensor_info(tf_example[preprocessor.FIELD_NAME]),
             model.lengths_key: tf.saved_model.utils.build_tensor_info(model.lengths)

        }
        sig_output = SignatureOutput(classes, values)
        sig_name = 'tag_text'
        assets = create_assets(model_file, sig_input, sig_output, sig_name, model.lengths_key)
        return sig_input, sig_output, sig_name, assets
