import tensorflow as tf
from collections import namedtuple


class SignatureInput(object):
    BASE_TENSOR_INPUT_NAMES = ['x', 'xch']

    CLASSIFY_INPUT_KEY = tf.saved_model.signature_constants.CLASSIFY_INPUTS
    PREDICT_INPUT_KEY = 'tokens'

    def __init__(self, classify=None, predict=None, extra_features=[], model=None):
        """
        accepts the preprocessed tensors for classify and predict along with
        a list of extra features that a model may define.

        Because we allow random extra features, we inspect the model for
        the related tensor. this is only currently supported for the predict
        endpoint.
        """
        self.input_list = SignatureInput.BASE_TENSOR_INPUT_NAMES + extra_features

        self.extra_features = extra_features

        if classify is not None:
            self.classify_sig = self._create_classify_sig(classify)

        if predict is None and model is not None:
            self.predict_sig = self._build_predict_signature_from_model(model)
        elif predict is not None:
            self.predict_sig = self._create_predict_sig(predict)

    @property
    def classify(self):
        return self.classify_sig or {}

    def _create_classify_sig(self, tensor):
        res = {
            SignatureInput.CLASSIFY_INPUT_KEY: 
                tf.saved_model.utils.build_tensor_info(tensor)
        }
        return res

    @property
    def predict(self):
        return self.predict_sig

    def _create_predict_sig(self, tensor):
        res = {}
        raw_post = tensor['text/tokens']
        res['tokens'] = tf.saved_model.utils.build_tensor_info(raw_post)

        for extra in self.extra_features:
            raw = tensor[extra]
            res[extra] = tf.saved_model.utils.build_tensor_info(raw)
        return res

    def _build_predict_signature_from_model(self, model):
        predict_tensors = {}

        for v in self.input_list:
            try:
                val = getattr(model, v)
                predict_tensors[v] = tf.saved_model.utils.build_tensor_info(val)
            except:
                pass # ugh

        return predict_tensors

SignatureOutput = namedtuple("SignatureOutput", ("classes", "scores"))
