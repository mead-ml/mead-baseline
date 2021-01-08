import os
import logging
import torch
import torch.nn as nn
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
from baseline.vectorizers import (
    GOVectorizer,
    Dict1DVectorizer,
    Char2DVectorizer,
    Dict2DVectorizer,
    Char1DVectorizer,
    Token1DVectorizer,
)
from mead.utils import (
    get_output_paths,
    create_metadata,
    save_to_bundle,
)
from mead.exporters import Exporter, register_exporter

__all__ = []
export = exporter(__all__)
logger = logging.getLogger('mead')


VECTORIZER_SHAPE_MAP = {
    Token1DVectorizer: [1, 100],
    GOVectorizer: [1, 100],
    Dict1DVectorizer: [1, 100],
    Char2DVectorizer: [1, 100, 50],
    Dict2DVectorizer: [1, 100, 50],
    Char1DVectorizer: [1, 100],
}

FAKE_SENTENCE = """
Common starlings may be kept as pets or as laboratory animals . Austrian <unk> Konrad Lorenz wrote of them in his book King Solomon 's Ring as " the poor man 's dog " and " something to love " , because nestlings are easily obtained from the wild and after careful hand rearing they are straightforward to look after . They adapt well to captivity , and thrive on a diet of standard bird feed and <unk> . Several birds may be kept in the same cage , and their <unk> makes them easy to train or study . The only disadvantages are their <unk> and indiscriminate defecation habits and the need to take precautions against diseases that may be transmitted to humans . As a laboratory bird , the common starling is second in numbers only to the domestic <unk> . 
"""

def create_fake_data(shapes, vectorizers, order, min_=0, max_=50,):
    data = {
        k: torch.randint(min_, max_, shapes[type(v)]) for k, v in vectorizers.items()
    }
    ordered_data = tuple(data[k] for k in order.embeddings)
    lengths = torch.LongTensor([data[list(data.keys())[0]].shape[1]])
    return ordered_data, lengths


def create_data_dict(vocabs, vectorizers, transpose=False, min_=0, max_=50, default_size=100):
    data = {}
    lengths = None
    for k, v in vectorizers.items():
        data[k], feature_length = vectorizers[k].run(FAKE_SENTENCE.split(), vocabs[k])
        data[k] = torch.LongTensor(data[k]).unsqueeze(0)
        if not lengths:
            lengths = [feature_length]
        # TODO: use the vectorizers, thats their job!!
        # data[k][0][0] = 101

    lengths = torch.LongTensor(lengths)

    if transpose:
        for k in vectorizers.keys():
            if len(data[k].shape) > 1:
                data[k] = data[k].transpose(0, 1)
    data['lengths'] = lengths
    return data


@export
class PytorchONNXExporter(Exporter):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.transpose = kwargs.get('transpose', False)
        self.tracing = kwargs.get('tracing', True)
        self.default_size = int(kwargs.get('default_size', 100))
        self.onnx_opset = int(kwargs.get('onnx_opset', 12))

    def apply_model_patches(self, model):
        return model

    def create_example_input(self, vocabs, vectorizers):
        return create_data_dict(vocabs, vectorizers, self.transpose, self.default_size)

    def create_example_output(self, model):
        return torch.ones((1, len(model.labels)))

    def create_model_inputs(self, model):
        return [k for k, _ in model.embeddings.items()] + ['lengths']

    def create_model_outputs(self, model):
        return ['output']

    def create_dynamic_axes(self, model, vectorizers, inputs, outputs):
        dynamics = {}#'output': {1: 'sequence'}}
        for name in outputs:
            dynamics[name] = {1: 'sequence'}
        for k, _ in model.embeddings.items():
            if k == 'char':
                dynamics[k] = {1: 'sequence', 2: 'chars'}
            else:
                dynamics[k] = {1: 'sequence'}
        return dynamics

    def _run(self, basename, output_dir, project=None, name=None, model_version=None, use_version=False, zip_results=True, **kwargs):
        logger.warning("Pytorch exporting is experimental and is not guaranteed to work for plugin models.")
        client_output, server_output = get_output_paths(
            output_dir,
            project, name,
            model_version,
            kwargs.get('remote', False),
            use_version=use_version
        )
        logger.info("Saving vectorizers and vocabs to %s", client_output)
        logger.info("Saving serialized model to %s", server_output)

        model, vectorizers, vocabs, model_name = self.load_model(basename)
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

        torch.onnx.export(model, data,
                          verbose=True,
                          dynamic_axes=dynamics,
                          f=f'{server_output}/{model_name}.onnx',
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


@export
@register_exporter(task='classify', name='default')
class ClassifyPytorchONNXExporter(PytorchONNXExporter):
    def __init__(self, task, **kwargs):
        super().__init__(task)
        self.sig_name = 'predict_text'


@export
@register_exporter(task='tagger', name='default')
class TaggerPytorchONNXExporter(PytorchONNXExporter):
    def __init__(self, task, **kwargs):
        super().__init__(task)
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
        super().__init__(task)
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
