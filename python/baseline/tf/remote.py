import numpy as np
from baseline.remote import RemoteModelREST, RemoteModelGRPC, register_remote


@register_remote('http')
class RemoteModelRESTTensorFlow(RemoteModelREST):

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


@register_remote('grpc')
class RemoteModelGRPCTensorFlow(RemoteModelGRPC): pass


@register_remote('grpc-preproc')
class RemoteModelGRPCTensorFlowPreproc(RemoteModelGRPCTensorFlow):

    def create_request(self, examples):
        request = self.predictpb.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = self.signature
        if self.version is not None:
            request.model_spec.version.value = self.version

        for key in examples:
            if key.endswith('lengths'):
                # assumed to be a np.array
                tensor_proto = self._make_proto(examples[key])
                request.inputs[key].CopyFrom(
                    tensor_proto
                )
            else:
                # Can be a np.array or a list
                request.inputs[key].CopyFrom(
                    self._make_proto(
                        examples[key],
                        dtype=type(examples[key][0]),
                        shape=[len(examples[key]), 1]
                    )
                )
        return request


@register_remote('http-preproc')
class RemoteModelRESTTensorFlowPreproc(RemoteModelRESTTensorFlow):

    def create_request(self, examples):
        inputs = {}
        if isinstance(examples['tokens'], np.ndarray):
            inputs['tokens'] = examples['tokens'].tolist()
        else:
            inputs['tokens'] = examples['tokens']
        for feature in self.input_keys:
            if feature.endswith('lengths'):
                if isinstance(examples[feature], np.ndarray):
                    inputs[feature] = examples[feature].tolist()
                else:
                    inputs[feature] = examples[feature]
        return {'signature_name': self.signature, 'inputs': inputs}
