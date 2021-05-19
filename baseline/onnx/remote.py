import numpy as np
from urllib.parse import urlparse
import json
from baseline.onnx.apis import predict_pb2, onnx_ml_pb2
from http.client import HTTPConnection

from baseline.remote import (
    RemoteModel,
    register_remote,
)


class RemoteONNXPredictModel(RemoteModel):
    def __init__(
            self,
            remote,
            name, signature,
            labels=None,
            beam=None,
            lengths_key=None,
            inputs=None,
            version=1,
            return_labels=None,
    ):
        """A remote model with REST transport

        :param remote: The remote endpoint
        :param name:  The name of the model
        :param signature: The model signature
        :param labels: The labels (defaults to None)
        :param beam: The beam width (defaults to None)
        :param lengths_key: Which key is used for the length of the input vector (defaults to None)
        :param inputs: The inputs (defaults to empty list)
        :param version: The model version (defaults to None)
        :param return_labels: Whether the remote model returns class indices or the class labels directly. This depends
        on the `return_labels` parameter in exporters
        """
        super().__init__(
            remote, name, signature, labels, beam, lengths_key, inputs, version, return_labels
        )
        url = urlparse(self.remote)
        if len(url.netloc.split(":")) != 2:
            raise ValueError("remote has to have the form <host_name>:<port>")
        self.hostname, self.port = url.netloc.split(":")
        #v_str = '/versions/{}:'.format(self.version) if self.version is not None else ''
        # TODO: this a hack
        v_str = 'versions/1:'
        path = url.path if url.path.endswith("/") else "{}/".format(url.path)
        self.path = f'{path}v1/models/{self.name}/{v_str}predict'
        self.headers = {'Content-type': 'application/x-protobuf',
                        'Accept': 'application/x-protobuf'}

    def predict(self, examples, **kwargs):
        """Run prediction over HTTP/REST.

        :param examples: The input examples
        :return: The outcomes
        """

        request = self.create_request(examples)
        conn = HTTPConnection(self.hostname, self.port)
        conn.request('POST', self.path, request.SerializeToString(), self.headers)
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f"remote server returns error: {response.reason}")
        response = response.read()
        response_message = predict_pb2.PredictResponse()
        response_message.ParseFromString(response)
        #if "error" in outcomes_list:
        #    raise ValueError("remote server returns error: {0}".format(outcomes_list["error"]))
        outcomes_list = response_message.outputs
        outcomes_list = self.deserialize_response(examples, outcomes_list)
        return outcomes_list

@register_remote('http-classify')
class RemoteONNXClassifier(RemoteONNXPredictModel):
    def deserialize_response(self, examples, predict_response):
        """Convert the response into a standard format."""
        labels = np.frombuffer(predict_response['output'].raw_data, dtype=np.float32)
        labels = [(np.array([i], np.int32), np.array([l], np.float32)) for i, l in enumerate(labels)]
        return [labels]


    def create_request(self, inputs):
        request_message = predict_pb2.PredictRequest()
        for k, v in inputs.items():
            # TODO: fix this hack
            k_name = 'lengths' if k.endswith('_lengths') else k
            if 'lengths' in k:
                continue
            input_tensor = onnx_ml_pb2.TensorProto()
            input_tensor.dims.extend(v.shape)
            input_tensor.data_type = 7
            input_tensor.raw_data = v.tobytes()
            request_message.inputs[k_name].data_type = input_tensor.data_type
            request_message.inputs[k_name].dims.extend(input_tensor.dims)
            request_message.inputs[k_name].raw_data = input_tensor.raw_data

        return request_message

@register_remote('http-tagger')
class RemoteONNXTagger(RemoteONNXPredictModel):
    def deserialize_response(self, examples, predict_response):
        """Convert the response into a standard format."""
        labels = np.frombuffer(predict_response['output'].raw_data, dtype=np.int64)

        return [labels]

    def create_request(self, inputs):
        request_message = predict_pb2.PredictRequest()
        have_lengths = False
        for k, v in inputs.items():
            if k.endswith('_lengths'):
                if have_lengths:
                    continue
                else:
                    k = 'lengths'
                    have_lengths = True
            input_tensor = onnx_ml_pb2.TensorProto()
            input_tensor.dims.extend(v.shape)
            input_tensor.data_type = 7
            input_tensor.raw_data = v.tobytes()
            request_message.inputs[k].data_type = input_tensor.data_type
            request_message.inputs[k].dims.extend(input_tensor.dims)
            request_message.inputs[k].raw_data = input_tensor.raw_data

        return request_message
