import os
import logging
import torch
import torch.nn as nn
import baseline as bl
from baseline.utils import (
    export,
    Offsets,
    write_json,
    load_vectorizers,
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
from mead.pytorch.tagger_decoders import InferenceCRF, InferenceGreedyDecoder


__all__ = []
exporter = export(__all__)
logger = logging.getLogger('mead')


VECTORIZER_SHAPE_MAP = {
    Token1DVectorizer: [1, 10],
    GOVectorizer: [1, 10],
    Dict1DVectorizer: [1, 10],
    Char2DVectorizer: [1, 10, 5],
    Dict2DVectorizer: [1, 10, 5],
    Char1DVectorizer: [1, 10],
}


def create_fake_data(shapes, vectorizers, order, min_=0, max_=50,):
    data = {
        k: torch.randint(min_, max_, shapes[type(v)]) for k, v in vectorizers.items()
    }
    ordered_data = tuple(data[k] for k in order)
    lengths = torch.LongTensor([data[list(data.keys())[0]].shape[1]])
    return ordered_data, lengths


def monkey_patch_embeddings(model):
    order = tuple(k for k, _ in model.embeddings.items())
    logger.debug("Using %s as the feature order", order)
    model.ordered_embeddings = tuple(model.embeddings[k] for k in order)

    def embed(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.ordered_embeddings[i](x[i]))
        return torch.cat(res, dim=2)

    model.embed = embed.__get__(model)
    return order


class ExportingTagger(nn.Module):
    def __init__(self, tagger):
        super(ExportingTagger, self).__init__()
        self.tagger = tagger
        if hasattr(tagger, 'crf'):
            logger.debug("Found CRF, replacing with torch script decoder.")
            self.decoder = InferenceCRF(
                self.tagger.crf.transitions.squeeze(0),
                self.tagger.crf.start_idx,
                self.tagger.crf.end_idx
            )
        else:
            if tagger.constraint is None:
                # This just calls torch.max, this is normally done in code for the tagger but we
                # wrap in a class here so that we can have a consistent forward.
                self.decoder = InferenceGreedyDecoder()
            else:
                logger.debug("Found constraints for decoding, replacing with torch script decoder.")
                self.decoder = InferenceCRF(
                    self.tagger.constraint.squeeze(0),
                    Offsets.GO,
                    Offsets.EOS
                )

    def forward(self, x, l):
        trans_x = []
        for i in range(len(x)):
            trans_x.append(x[i].transpose(0, 1))
        new_x = tuple(trans_x)
        x = self.tagger.compute_unaries(new_x, l)
        return self.decoder.decode(x, l)[0]


class ExportingClassifier(nn.Module):
    def __init__(self, classifier):
        super(ExportingClassifier, self).__init__()
        self.classifier = classifier

    def forward(self, x, l):
        x = self.classifier.embed(x)
        x = self.classifier.pool(x, l)
        x = self.classifier.stacked(x)
        return self.classifier.output(x)


@exporter
class PytorchExporter(Exporter):
    def __init__(self, task, **kwargs):
        super(PytorchExporter, self).__init__(task, **kwargs)
        self.wrapper = None


    def run(self, basename, output_dir, model_version, **kwargs):
        logger.warning("Pytorch exporting is experimental and is not guaranteed to work for plugin models.")
        client_output, server_output = get_output_paths(
            output_dir, model_version, kwargs.get('remote', True)
        )
        logger.info("Saving vectorizers and vocabs to %s", client_output)
        logger.info("Saving serialized model to %s", server_output)
        model, vectorizers, model_name = self.load_model(basename)
        order = monkey_patch_embeddings(model)
        data, lengths = create_fake_data(VECTORIZER_SHAPE_MAP, vectorizers, order)
        meta = create_metadata(order, ['output'], self.sig_name, model_name, model.lengths_key)

        exportable = self.wrapper(model)
        logger.info("Tracing Model.")
        traced = torch.jit.trace(exportable, (data, lengths))
        traced.save(os.path.join(server_output, 'model.pt'))

        logger.info("Saving metadata.")
        save_to_bundle(client_output, basename, assets=meta)
        logger.info('Successfully exported model to %s', output_dir)


    def load_model(self, model_dir):
        model_name = find_model_basename(model_dir)
        vectorizers = load_vectorizers(model_dir)
        model = load_model_for(self.task.task_name(), model_name, device='cpu')
        model = model.cpu()
        model.eval()
        model_name = os.path.basename(model_name)
        return model, vectorizers, model_name


@exporter
@register_exporter(task='classify', name='default')
class ClassifyPytorchExporter(PytorchExporter):
    def __init__(self, task, **kwargs):
        super(ClassifyPytorchExporter, self).__init__(task)
        self.wrapper = ExportingClassifier
        self.sig_name = 'predict_text'


@exporter
@register_exporter(task='tagger', name='default')
class TaggerPytorchExporter(PytorchExporter):
    def __init__(self, task, **kwargs):
        super(TaggerPytorchExporter, self).__init__(task)
        self.wrapper = ExportingTagger
        self.sig_name = 'tag_text'


@exporter
@register_exporter(task='seq2seq', name='default')
class Seq2SeqPytorchExporter(PytorchExporter):
    def __init__(self, task, **kwargs):
        raise NotImplementedError
