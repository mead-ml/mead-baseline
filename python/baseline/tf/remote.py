import numpy as np
from baseline.utils import import_user_module
import tensorflow as tf

class RemoteModelTensorFlow(object):
    def __init__(self, remote, name, signature, labels=None, beam=None, lengths_key=None, inputs=[]):
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
        return self.labels

    def predict(self, examples):
        valid_example = all(k in examples for k in self.input_keys)
        if not valid_example:
            raise ValueError("should have keys: " + ",".join(self.input_keys))

        request = self.create_request(examples)
        stub = self.servicepb.PredictionServiceStub(self.channel)
        outcomes_list = stub.Predict(request)
        outcomes_list = self.deserialize_response(examples, outcomes_list)

        return outcomes_list

    def create_request(self, examples):
        request = self.predictpb.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = self.signature

        for feature in self.input_keys:
            if isinstance(examples[feature], np.ndarray): 
                shape = examples[feature].shape
            else:
                shape = [1]

            import tensorflow
            tensor_proto = tensorflow.contrib.util.make_tensor_proto(examples[feature], shape=shape, dtype=tf.int32)
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
            for i in range(num_ex):
                length = len(self.get_labels())
                d = [(c,s) for c,s in zip(classes[length*i:length*(i+1)], scores[length*i:length*(i+1)])]
                result.append(d)
            
            return result