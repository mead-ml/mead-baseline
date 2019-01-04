import numpy as np
from baseline.utils import import_user_module


class RemoteModelTensorFlowREST(object):

    def __init__(self, remote, name, signature, labels=None, beam=None, lengths_key=None, inputs=[], version=None):
        """A remote model that lives on TF serving with REST transport

        This type of model currently depends on the `requests` module as a dependency for HTTP

        :param remote: The remote endpoint
        :param name:  The name of the model
        :param signature: The model signature
        :param labels: The labels (defaults to None)
        :param beam: The beam width (defaults to None)
        :param lengths_key: Which key is used for the length of the input vector (defaults to None)
        :param inputs: The inputs (defaults to empty list)
        :param version: The model version (defaults to None)
        """
        self.remote = remote
        self.name = name
        self.signature = signature
        self.lengths_key = lengths_key
        self.input_keys = set(inputs)
        self.beam = beam
        self.labels = labels
        self.version = version

    def get_labels(self):
        """Return the model's labels

        :return: The model's labels
        """
        return self.labels

    def predict(self, examples):
        """Run prediction over HTTP/REST.

        :param examples: The input examples
        :return: The outcomes
        """
        import requests

        valid_example = all(k in examples for k in self.input_keys)
        if not valid_example:
            raise ValueError("should have keys: " + ",".join(self.input_keys))

        v_str = '/versions/{}'.format(self.version) if self.version is not None else ''
        url = '{}/v1/models/{}{}:predict'.format(self.remote, self.name, v_str)
        request = self.create_request(examples)
        outcomes_list = requests.post(url, json=request)
        outcomes_list = outcomes_list.json()
        if "error" in outcomes_list:
            raise ValueError("remote server returns error: {0}".format(outcomes_list["error"]))
        outcomes_list = outcomes_list["outputs"]
        outcomes_list = self.deserialize_response(examples, outcomes_list)
        return outcomes_list

    def deserialize_response(self, examples, predict_response):
        """
        read the JSON response from tensorflow serving and decode it according
        to the signature.
        :param examples: Input examples
        :param predict_response: an HTTP/REST output
        """
        if self.signature == 'suggest_text':
            # s2s returns int values.
            tensor = np.array(predict_response)
            tensor = tensor.transpose(1, 2, 0)
            return tensor

        if self.signature == 'tag_text':
            classes = predict_response['classes']
            lengths = examples[self.lengths_key]
            result = []
            num_ex = examples[self.lengths_key].shape[0]
            for i in range(num_ex):
                length_i = lengths[i]
                classes_i = classes[i]
                d = [classes_i[j] for j in range(length_i)]
                result.append(d)

            return result

        if self.signature == 'predict_text':
            num_ex = len(examples[self.lengths_key])
            scores = predict_response['scores']
            classes = predict_response['classes']
            result = []
            for i in range(num_ex):
                score_i = scores[i]
                classes_i = classes[i]
                d = [(c, s) for c, s in zip(classes_i, score_i)]
                result.append(d)
            return result

    def create_request(self, examples):
        inputs = {}
        for feature in self.input_keys:
            tensor = examples[feature]
            if isinstance(tensor, np.ndarray):
                inputs[feature] = tensor.tolist()
            else:
                inputs[feature] = tensor
        request = {'signature_name': self.signature, 'inputs': inputs}
        return request


class RemoteModelTensorFlowGRPC(object):

    def __init__(self, remote, name, signature, labels=None, beam=None, lengths_key=None, inputs=[]):
        """A remote model that lives on TF serving with gRPC transport

        When using this type of model, there is an external dependency on the `grpc` package, as well as the
        TF serving protobuf stub files.  There is also currently a dependency on `tensorflow`

        :param remote: The remote endpoint
        :param name:  The name of the model
        :param signature: The model signature
        :param labels: The labels (defaults to None)
        :param beam: The beam width (defaults to None)
        :param lengths_key: Which key is used for the length of the input vector (defaults to None)
        :param inputs: The inputs (defaults to empty list)
        """
        self.predictpb = import_user_module('tensorflow_serving.apis.predict_pb2')
        self.servicepb = import_user_module('tensorflow_serving.apis.prediction_service_pb2_grpc')
        self.metadatapb = import_user_module('tensorflow_serving.apis.get_model_metadata_pb2')
        self.grpc = import_user_module('grpc')
        
        self.remote = remote
        self.name = name
        self.signature = signature

        self.channel = self.grpc.insecure_channel(remote)

        self.lengths_key = lengths_key
        self.input_keys = set(inputs)
        self.beam = beam
        self.labels = labels

    def get_labels(self):
        """Return the model's labels

        :return: The model's labels
        """
        return self.labels

    def predict(self, examples):
        """Run prediction over gRPC

        :param examples: The input examples
        :return: The outcomes
        """
        valid_example = all(k in examples for k in self.input_keys)
        if not valid_example:
            raise ValueError("should have keys: " + ",".join(self.input_keys))

        request = self.create_request(examples)
        stub = self.servicepb.PredictionServiceStub(self.channel)
        outcomes_list = stub.Predict(request)
        outcomes_list = self.deserialize_response(examples, outcomes_list)

        return outcomes_list

    def create_request(self, examples):
        # TODO: Remove TF dependency client side
        import tensorflow as tf

        request = self.predictpb.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = self.signature

        for feature in self.input_keys:
            if isinstance(examples[feature], np.ndarray): 
                shape = examples[feature].shape
            else:
                shape = [1]

            tensor_proto = tf.contrib.util.make_tensor_proto(examples[feature], shape=shape, dtype=tf.int32)
            request.inputs[feature].CopyFrom(
                tensor_proto
            )

        return request

    def deserialize_response(self, examples, predict_response):
        """
        read the protobuf response from tensorflow serving and decode it according
        to the signature.

        here's the relevant piece of the proto:
            map<string, TensorProto> inputs = 2;

        the predict endpoint happens to have the ability to filter output for certain keys, but
        we do not support this currently. There are two keys we want to extract: classes and scores.

        :params predict_response: a PredictResponse protobuf object, 
                    as defined in tensorflow_serving proto files
        """
        if self.signature == 'suggest_text':
            # s2s returns int values.
            classes = predict_response.outputs.get('classes').int_val
            results = [classes[x:x+self.beam] for x in range(0, len(classes), self.beam)]
            results = list(zip(*results)) #transpose
            return [results]

        if self.signature == 'tag_text':
            classes = predict_response.outputs.get('classes').int_val
            lengths = examples[self.lengths_key]
            result = []
            for i in range(examples[self.lengths_key].shape[0]):
                length = lengths[i]
                result.append([np.int32(x) for x in classes[length*i:length*(i+1)]])
            
            return result
            
        if self.signature == 'predict_text':
            scores = predict_response.outputs.get('scores').float_val
            classes = predict_response.outputs.get('classes').string_val
            result = []
            num_ex = len(examples[self.lengths_key])
            length = len(self.get_labels())
            for i in range(num_ex):
                d = [(c, s) for c, s in zip(classes[length*i:length*(i+1)], scores[length*i:length*(i+1)])]
                result.append(d)
            
            return result
