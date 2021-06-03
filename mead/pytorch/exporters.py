import os
import logging
import torch
import torch.nn as nn
from typing import Dict
import baseline as bl
from eight_mile.pytorch.layers import (
    CRF,
    ViterbiBatchSize1,
    TaggerGreedyDecoder,
    ViterbiLogSoftmaxNormBatchSize1
)
from baseline.utils import (
    exporter,
    Offsets,
    write_json,
    load_vectorizers,
    load_vocabs,
    find_model_basename,
)
from baseline.model import load_model_for

from mead.utils import (
    get_output_paths,
    create_metadata,
    save_to_bundle,
)
from mead.exporters import Exporter, register_exporter

__all__ = []
export = exporter(__all__)
logger = logging.getLogger('mead')

REMOTE_MODEL_NAME = 'model'

S1D = """
Common starlings may be kept as pets or as laboratory animals . Austrian <unk> Konrad Lorenz wrote of them in his book King Solomon 's Ring as " the poor man 's dog " and " something to love " , because nestlings are easily obtained from the wild and after careful hand rearing they are straightforward to look after . They adapt well to captivity , and thrive on a diet of standard bird feed and <unk> . Several birds may be kept in the same cage , and their <unk> makes them easy to train or study . The only disadvantages are their <unk> and indiscriminate defecation habits and the need to take precautions against diseases that may be transmitted to humans . As a laboratory bird , the common starling is second in numbers only to the domestic <unk> . 
"""

S2D = [
  "Common starlings may be kept as pets or as laboratory animals .",
  "Austrian <unk> Konrad Lorenz wrote of them in his book King Solomon 's Ring as \" the poor man 's dog \" and \" something to love \" , because nestlings are easily obtained from the wild and after careful hand rearing they are straightforward to look after . ",
  "They adapt well to captivity , and thrive on a diet of standard bird feed and <unk> . ",
  "The only disadvantages are their <unk> and indiscriminate defecation habits and the need to take precautions against diseases that may be transmitted to humans . ",
  "As a laboratory bird , the common starling is second in numbers only to the domestic <unk> ."
]


def create_data_dict(vocabs, vectorizers, transpose=False):
    data = {}
    lengths = None
    for k, v in vectorizers.items():
        data[k], feature_length = vectorizers[k].run(S1D.split(), vocabs[k])
        data[k] = torch.LongTensor(data[k]).unsqueeze(0)

        if not lengths:
            lengths = [feature_length]

    lengths = torch.LongTensor(lengths)

    if transpose:
        for k in vectorizers.keys():
            if len(data[k].shape) > 1:
                data[k] = data[k].transpose(0, 1)
    data['lengths'] = lengths
    return data


def create_data_dict_nbest(vocabs, vectorizers):
    data = {}

    length_tensor = None
    for k, v in vectorizers.items():
        lengths = []
        vectors = []
        for sentence in S2D:
            vec, feature_length = vectorizers[k].run(sentence.split(), vocabs[k])
            vectors.append(torch.LongTensor(vec))
            lengths.append(torch.LongTensor([feature_length]))
        data[k] = torch.stack(vectors).unsqueeze(0)

        if not length_tensor:
            length_tensor = torch.stack(lengths).reshape(1, -1)

    data['lengths'] = length_tensor
    return data


@export
class PytorchONNXExporter(Exporter):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.transpose = kwargs.get('transpose', False)
        self.tracing = kwargs.get('tracing', True)
        self.default_size = int(kwargs.get('default_size', 100))
        self.onnx_opset = int(kwargs.get('onnx_opset', 12))
        self.nbest_inputs = bool(kwargs.get('nbest_input', False))

    def apply_model_patches(self, model):
        return model

    def create_example_input(self, vocabs, vectorizers):
        if self.nbest_inputs:
            return create_data_dict_nbest(vocabs, vectorizers)
        return create_data_dict(vocabs, vectorizers, self.transpose)

    def create_example_output(self, model):
        if hasattr(model, 'output'):
            if isinstance(model.output, nn.ModuleList):
                return [torch.ones((1, len(model.labels[i][1]))) for i in range(len(model.output))]
        return torch.ones((1, len(model.labels)))

    def create_model_inputs(self, model):
        return [k for k in model.embeddings.keys()] + ['lengths']

    def create_model_outputs(self, model):
        if hasattr(model, 'output'):
            if isinstance(model.output, nn.ModuleList):
                logger.info("Multiheaded model")
                return [f"output_{i}" for i in range(len(model.output))]
        return ['output']

    def create_dynamic_axes(self, model, vectorizers, inputs, outputs):
        dynamics = {}
        for name in outputs:
            dynamics[name] = {1: 'sequence'}
        for k in model.embeddings.keys():
            if k == 'char':
                dynamics[k] = {1: 'sequence', 2: 'chars'}
            else:
                dynamics[k] = {1: 'sequence'}

        if self.nbest_inputs:
            for name in inputs:
                if 'lengths' == name:
                    dynamics[name] = {1: 'sequence'}
                else:
                    dynamics[name] = {1: 'nbest', 2: 'sequence'}
        return dynamics

    def _run(self, basename, output_dir, project=None, name=None, model_version=None, use_version=False, zip_results=True,
             remote=False, **kwargs):
        client_output, server_output = get_output_paths(
            output_dir,
            project, name,
            model_version,
            remote,
            use_version=use_version
        )
        logger.info("Saving vectorizers and vocabs to %s", client_output)
        logger.info("Saving serialized model to %s", server_output)

        model, vectorizers, vocabs, model_name = self.load_model(basename)
        # Triton server wants to see a specific name

        model = self.apply_model_patches(model)

        data = self.create_example_input(vocabs, vectorizers)
        example_output = self.create_example_output(model)

        inputs = self.create_model_inputs(model)
        outputs = self.create_model_outputs(model)

        dynamics = self.create_dynamic_axes(model, vectorizers, inputs, outputs)

        meta = create_metadata(
            inputs, outputs,
            self.sig_name,
            model_name, model.lengths_key
        )

        if not self.tracing:
            model = torch.jit.script(model)

        logger.info("Exporting Model.")
        logger.info("Model inputs: %s", inputs)
        logger.info("Model outputs: %s", outputs)

        onnx_model_name = REMOTE_MODEL_NAME if remote else model_name

        torch.onnx.export(model, data,
                          verbose=True,
                          dynamic_axes=dynamics,
                          f=f'{server_output}/{onnx_model_name}.onnx',
                          input_names=inputs,
                          output_names=outputs,
                          opset_version=self.onnx_opset,
                          #propagate=True,
                          example_outputs=example_output)

        logger.info("Saving metadata.")
        save_to_bundle(client_output, basename, assets=meta, zip_results=zip_results)
        logger.info('Successfully exported model to %s', output_dir)
        return client_output, server_output

    def load_model(self, model_dir):
        model_name = find_model_basename(model_dir)
        vectorizers = load_vectorizers(model_dir)
        vocabs = load_vocabs(model_dir)
        model = load_model_for(self.task.task_name(), model_name, device='cpu')
        model = model.cpu()
        model.eval()
        model_name = os.path.basename(model_name)
        return model, vectorizers, vocabs, model_name


class Embedder(nn.ModuleList):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        return self.target.embed(inputs)

    @property
    def embeddings(self):
        return self.target.embeddings

    @property
    def lengths_key(self):
        return self.target.lengths_key

    @property
    def embed_output_dim(self):
        return self.target.embed_output_dim

@export
@register_exporter(task='classify', name='embed')
class EmbedPytorchONNXExporter(PytorchONNXExporter):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.sig_name = 'embed_text'

    def load_model(self, model_dir):
        model_name = find_model_basename(model_dir)
        vectorizers = load_vectorizers(model_dir)
        vocabs = load_vocabs(model_dir)
        model = load_model_for(self.task.task_name(), model_name, device='cpu')
        model = Embedder(model)
        model = model.cpu()
        model.eval()
        model_name = os.path.basename(model_name)
        return model, vectorizers, vocabs, model_name

    def create_example_output(self, model):
        return torch.ones((1, model.embed_output_dim), dtype=torch.float32)

@export
@register_exporter(task='classify', name='default')
class ClassifyPytorchONNXExporter(PytorchONNXExporter):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.sig_name = 'predict_text'


@export
@register_exporter(task='tagger', name='default')
class TaggerPytorchONNXExporter(PytorchONNXExporter):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.sig_name = 'tag_text'

    def apply_model_patches(self, model):
        if hasattr(model, 'decoder'):
            if isinstance(model.decoder, CRF):
                model.decoder.viterbi = ViterbiBatchSize1(model.decoder.viterbi.start_idx,
                                                          model.decoder.viterbi.end_idx)
            elif isinstance(model.decoder, TaggerGreedyDecoder):
                model.decoder.viterbi = ViterbiLogSoftmaxNormBatchSize1(
                    model.decoder.viterbi.start_idx,
                    model.decoder.viterbi.end_idx
                )
        return model


@export
@register_exporter(task='deps', name='default')
class DependencyParserPytorchONNXExporter(PytorchONNXExporter):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.sig_name = 'deps_text'

    def create_example_output(self, model):
        return torch.ones(1, self.default_size, self.default_size), torch.ones(1, self.default_size, len(model.labels))

    def create_model_outputs(self, model):
        return ['arcs', 'labels']

    def apply_model_patches(self, model):
        for _, e in model.embeddings.items():
            #  Turn off dropin flag, unsupported
            # https://github.com/pytorch/pytorch/issues/49001
            if hasattr(e, 'dropin'):
                e.dropin = 0
        return model
