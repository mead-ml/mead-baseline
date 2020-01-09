import numpy as np
from baseline.remote import (
    RemoteModelREST,
    RemoteModelGRPC,
    register_remote,
    RemoteRESTClassifier,
    RemoteRESTTagger,
    RemoteRESTSeq2Seq,
    RemoteRESTEmbeddings,
    RemoteGRPCClassifier,
    RemoteGRPCTagger,
    RemoteGRPCSeq2Seq,
    RemoteGRPCEmbeddings,
)


class RemoteRESTTensorFlowMixin(RemoteModelREST):

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

@register_remote('http-classify')
class RemoteRESTTensorFlowClassifier(RemoteRESTTensorFlowMixin, RemoteRESTClassifier): pass

@register_remote('http-tagger')
class RemoteRESTTensorFlowTagger(RemoteRESTTensorFlowMixin, RemoteRESTTagger): pass

@register_remote('http-seq2seq')
class RemoteRESTTensorFlowSeq2Seq(RemoteRESTTensorFlowMixin, RemoteRESTSeq2Seq): pass

@register_remote('http-servable-embeddings')
class RemoteRESTTensorFlowEmbeddings(RemoteRESTTensorFlowMixin, RemoteRESTEmbeddings): pass


@register_remote('grpc')
class RemoteGRPCTensorFlowMixin(RemoteModelGRPC): pass

@register_remote('grpc-classify')
class RemoteGRPCTensorFlowClassifier(RemoteGRPCTensorFlowMixin, RemoteGRPCClassifier): pass

@register_remote('grpc-tagger')
class RemoteGRPCTensorFlowTagger(RemoteGRPCTensorFlowMixin, RemoteGRPCTagger): pass

@register_remote('grpc-seq2seq')
class RemoteGRPCTensorFlowSeq2Seq(RemoteGRPCTensorFlowMixin, RemoteGRPCSeq2Seq): pass

@register_remote('grpc-servable-embeddings')
class RemoteGRPCTensorFlowEmbeddings(RemoteGRPCTensorFlowMixin, RemoteGRPCEmbeddings): pass


@register_remote('grpc-preproc')
class RemoteGRPCTensorFlowPreprocMixin(RemoteModelGRPC):

    def create_request(self, examples):
        # TODO: Remove TF dependency client side
        import tensorflow as tf

        request = self.predictpb.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = self.signature
        if self.version is not None:
            request.model_spec.version.value = self.version

        for key in examples:
            if key.endswith('lengths'):
                shape = examples[key].shape
                tensor_proto = tf.contrib.util.make_tensor_proto(examples[key], shape=shape, dtype=tf.int32)
                request.inputs[key].CopyFrom(
                    tensor_proto
                )
            else:
                request.inputs[key].CopyFrom(
                    tf.contrib.util.make_tensor_proto(examples[key], shape=[len(examples[key]), 1])
                )
        return request

@register_remote('grpc-preproc-classify')
class RemoteGRPCTensorFlowPreprocClassifier(RemoteGRPCTensorFlowPreprocMixin, RemoteGRPCClassifier): pass

@register_remote('grpc-preproc-tagger')
class RemoteGRPCTensorFlowPreprocTagger(RemoteGRPCTensorFlowPreprocMixin, RemoteGRPCTagger): pass

@register_remote('grpc-preproc-seq2seq')
class RemoteGRPCTensorFlowPreprocSeq2Seq(RemoteGRPCTensorFlowPreprocMixin, RemoteGRPCSeq2Seq): pass

@register_remote('grpc-preproc-servable-embeddings')
class RemoteGRPCTensorFlowPreprocEmbeddings(RemoteGRPCTensorFlowPreprocMixin, RemoteGRPCEmbeddings): pass


@register_remote('http-preproc')
class RemoteRESTTensorFlowPreprocMixin(RemoteModelREST):

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

@register_remote('http-preproc-classify')
class RemoteRESTTensorFlowPreprocClassifier(RemoteRESTTensorFlowPreprocMixin, RemoteRESTClassifier): pass

@register_remote('http-preproc-tagger')
class RemoteRESTTensorFlowPreprocTagger(RemoteRESTTensorFlowPreprocMixin, RemoteRESTTagger): pass

@register_remote('http-preproc-seq2seq')
class RemoteRESTTensorFlowPreprocSeq2Seq(RemoteRESTTensorFlowPreprocMixin, RemoteRESTSeq2Seq): pass

@register_remote('http-preproc-servable-embeddings')
class RemoteRESTTensorFlowPreprocEmbeddings(RemoteRESTTensorFlowPreprocMixin, RemoteRESTEmbeddings): pass
