import numpy as np
import logging
from urllib.parse import urlparse
import json
from baseline.onnx.apis import model_config_pb2, grpc_service_pb2, grpc_service_pb2_grpc
#from http.client import HTTPConnection
import grpc

from baseline.remote import (
    RemoteModel,
    register_remote,
)
logger = logging.getLogger('baseline')


class RemoteONNXPredictModel(RemoteModel):
    def __init__(
            self,
            remote,
            name,
            signature,
            labels=None,
            beam=None,
            lengths_key=None,
            inputs=None,
            version=1,
            return_labels=None,
    ):
        """A remote model with gRPC transport

        :param remote: The remote endpoint (eg. 0.0.0.0:8000)
        :param name:  The name of the model (this is the same name as the sub-dir under the triton model dir
        :param signature: The model signature (this is deprecated)
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
        if len(self.remote.split(":")) != 2:
            raise ValueError("remote has to have the form <host_name>:<port>")

        channel = grpc.insecure_channel(self.remote)
        self.grpc_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel)
        self.version = version
        request = grpc_service_pb2.ModelMetadataRequest(name=self.name, version=self.version)



        response = self.grpc_stub.ModelMetadata(request)
        self.inputs = {i.name: i.datatype for i in response.inputs}
        self.outputs = {o.name: o.datatype for o in response.outputs}
        if self.version is None:
            self.version = response.versions[-1]
            logger.info("Selected version %s of model %s", self.version, self.name)

    def predict(self, examples, **kwargs):
        """Run prediction over HTTP/REST.

        :param examples: The input examples
        :return: The outcomes
        """

        request = self.create_request(examples)
        response = self.grpc_stub.ModelInfer(request)
        response = self.deserialize_response(examples, response)
        return response

    def create_request(self, inputs):

        request = grpc_service_pb2.ModelInferRequest()
        request.model_name = self.name
        request.model_version = str(self.version)
        request.id = "ID"  # ???
        require_length = False
        have_length = False
        if 'lengths' in self.inputs:
            require_length = True
        for k, v in inputs.items():
            if k not in self.inputs and not k.endswith('_lengths'):
                logger.warning("Unexpected input: %s, ignoring", k)

            if k.endswith('_lengths'):
                if k not in self.inputs:
                    if not require_length or have_length:
                        continue
                    else:
                        k = 'lengths'
                        have_length = True
            input = grpc_service_pb2.ModelInferRequest().InferInputTensor()
            input.name = k
            input.shape.extend(v.shape)
            input.datatype = self.inputs[k]
            request.inputs.extend([input])
            request.raw_input_contents.extend([v.tobytes()])

        for k, v in self.outputs.items():
            output = grpc_service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            output.name = k
            request.outputs.extend([output])

        return request

@register_remote('grpc-classify')
class RemoteONNXClassifier(RemoteONNXPredictModel):
    def deserialize_response(self, examples, predict_response):
        """Convert the response into a standard format."""
        label_values = np.frombuffer(predict_response.raw_output_contents[0], dtype=np.float32)
        labels = [(np.array([i], np.int32), np.array([l], np.float32)) for i, l in enumerate(label_values)]
        return [labels]

@register_remote('grpc-tagger')
class RemoteONNXTagger(RemoteONNXPredictModel):
    def deserialize_response(self, examples, predict_response):
        """Convert the response into a standard format."""
        label_values = np.frombuffer(predict_response.raw_output_contents[0], dtype=np.int64)

        return [label_values]
