import numpy as np
from baseline.remote import RemoteModelREST, RemoteModelGRPC, register_remote


def _convert(data):
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


@register_remote('http')
class RemoteModelRESTPytorch(RemoteModelREST):
    """JSON schema:

        {
            "signature_name": "name",
            "inputs": {
                "data": [
                    [...],
                    [...],
                    ...
                ],
                "shapes": [
                    [...],
                    [...],
                    ...
                ]
                "lengths": [...]
            }
        }

        data should be flattened (np.ravel)
        shapes should be the call needed to reshape the tensor (should be same size as data)

    """
    def predict(self, examples, **kwargs):
        """The pytorch server can only handle batch size of 1 because the JIT'd
        `pack_padded_sequence jits that batch size. So we send a request per
        example.
        """
        results = []
        example_input = examples[self.input_keys[0]]
        batch_size = len(example_input)
        for i in range(batch_size):
            example = {k: np.array([v[i]]) for k, v in examples.items()}
            example_output = super().predict(example, **kwargs)
            results.append(example_output[0])
        return results

    def create_request(self, examples):
        request = {}
        request['signature_name'] = self.signature
        request['inputs'] = {}
        request['inputs']['data'] = [_convert(examples[x]).ravel().tolist() for x in self.input_keys]
        request['inputs']['shapes'] = [list(examples[x].shape) for x in self.input_keys]
        request['inputs']['lengths'] = examples[self.lengths_key].tolist()
        return request


@register_remote('grpc')
class RemoteModelGRPCPytorch(RemoteModelGRPC):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Pytorch GRPC service is not implemented.')


@register_remote('grpc-preproc')
class RemoteModelGRPCPytorchPreproc(RemoteModelGRPCPytorch):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Pytorch does not support string tensors so Server side preproc is not supported.')


@register_remote('http-preproc')
class RemoteModelHTTPPytorchPreproc(RemoteModelRESTPytorch):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Pytorch does not support string tensors so Server side preproc is not supported.')
