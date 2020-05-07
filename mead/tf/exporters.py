import os
import shutil
import logging
import datetime
from collections import namedtuple
import baseline
from baseline.tf.embeddings import *
from baseline.utils import (
    exporter,
    Offsets,
    ls_props,
    read_json,
    write_json,
    transition_mask
)
from baseline.model import load_tagger_model, load_model, load_seq2seq_model
from mead.exporters import Exporter, register_exporter
from mead.utils import save_to_bundle, create_metadata, get_output_paths


__all__ = []
export = exporter(__all__)


FIELD_NAME = 'text/tokens'
ASSET_FILE_NAME = 'model.assets'
logger = logging.getLogger('mead')
SignatureOutput = namedtuple("SignatureOutput", ("classes", "scores"))


def get_tf_index_from_unzipped(dir_path):
    return os.path.join(dir_path, [x[:-6] for x in os.listdir(dir_path) if 'index' in x][0])


@export
class TensorFlowExporter(Exporter):

    def __init__(self, task, **kwargs):
        super(TensorFlowExporter, self).__init__(task, **kwargs)

    def _run(self, basename, output_dir, project=None, name=None, model_version=None, **kwargs):
        basename = get_tf_index_from_unzipped(basename)

        with tf.compat.v1.Graph().as_default():
            config_proto = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            with tf.compat.v1.Session(config=config_proto) as sess:
                sig_input, sig_output, sig_name, assets = self._create_rpc_call(sess, basename)
                client_output, server_output = get_output_paths(
                    output_dir,
                    project, name,
                    model_version,
                    kwargs.get('remote', True), make_server=False
                )
                logger.info('Saving vectorizers and vocabs to %s', client_output)
                logger.info('Saving serialized model to %s' % server_output)
                try:
                    builder = self._create_saved_model_builder(sess, server_output, sig_input, sig_output, sig_name)
                    create_bundle(builder, client_output, basename, assets)
                    logger.info('Successfully exported model to %s' % output_dir)
                except AssertionError as e:
                    # model already exists
                    raise e
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    # export process broke.
                    # TODO(MB): we should remove the directory, if one has been saved already.
                    raise e
        return client_output, server_output

    def _create_saved_model_builder(self, sess, output_path, sig_input, sig_output, sig_name):
        """
        create the SavedModelBuilder with standard endpoints.

        we reuse the classify constants from tensorflow to define the predict
        endpoint so that we can call the output by classes/scores.
        """
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(output_path)

        classes_output_tensor = tf.compat.v1.saved_model.utils.build_tensor_info(
            sig_output.classes)

        output_def_map = {
            tf.compat.v1.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classes_output_tensor
        }
        if sig_output.scores is not None:
            scores_output_tensor = tf.compat.v1.saved_model.utils.build_tensor_info(sig_output.scores)
            output_def_map[tf.compat.v1.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES] = scores_output_tensor

        prediction_signature = (
            tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs=sig_input,
                outputs=output_def_map,  # we reuse classify constants here.
                method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        legacy_init_op = tf.group(tf.compat.v1.tables_initializer(), name='legacy_init_op')
        definition = dict({})
        definition[sig_name] = prediction_signature
        builder.add_meta_graph_and_variables(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map=definition,
            legacy_init_op=legacy_init_op)

        return builder

    def _create_rpc_call(self, sess, basename, **kwargs):
        pass


@export
@register_exporter(task='classify', name='default')
class ClassifyTensorFlowExporter(TensorFlowExporter):

    def __init__(self, task, **kwargs):
        super(ClassifyTensorFlowExporter, self).__init__(task, **kwargs)
        self.return_labels = kwargs.get('return_labels', True)

    def _create_model(self, sess, basename, **kwargs):
        model = load_model(basename, sess=sess, **kwargs)
        values, indices = tf.nn.top_k(model.probs, len(model.labels))
        # Restore the checkpoint
        if self.return_labels:
            class_tensor = tf.constant(model.labels)
            table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
            classes = table.lookup(tf.to_int64(indices))
            return model, classes, values
        else:
            return model, indices, values

    def _create_rpc_call(self, sess, basename, **kwargs):
        model, classes, values = self._create_model(sess, basename)

        predict_tensors = {}
        predict_tensors[model.lengths_key] = tf.compat.v1.saved_model.utils.build_tensor_info(model.lengths)

        for k, v in model.embeddings.items():
            try:
                predict_tensors[k] = tf.compat.v1.saved_model.utils.build_tensor_info(v.x)
            except:
                raise Exception('Unknown attribute in signature: {}'.format(v))

        sig_input = predict_tensors
        sig_output = SignatureOutput(classes, values)
        sig_name = 'predict_text'

        assets = create_assets(
            basename,
            sig_input, sig_output, sig_name,
            model.lengths_key,
            return_labels=self.return_labels,
            preproc=self.preproc_type()
        )
        return sig_input, sig_output, sig_name, assets


@export
@register_exporter(task='tagger', name='default')
class TaggerTensorFlowExporter(TensorFlowExporter):

    def __init__(self, task, **kwargs):
        super(TaggerTensorFlowExporter, self).__init__(task, **kwargs)
        self.return_labels = kwargs.get('return_labels', False)  # keep default behavior

    def _create_model(self, sess, basename, **kwargs):
        model = load_tagger_model(basename, sess=sess, **kwargs)
        softmax_output = tf.nn.softmax(model.probs)
        values, _ = tf.nn.top_k(softmax_output, 1)
        indices = model.best
        if self.return_labels:
            labels = read_json(basename + '.labels')
            list_of_labels = [''] * len(labels)
            for label, idval in labels.items():
                list_of_labels[idval] = label
            class_tensor = tf.constant(list_of_labels)
            table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
            classes = table.lookup(tf.to_int64(indices))
            return model, classes, values
        else:
            return model, indices, values

    def _create_rpc_call(self, sess, basename, **kwargs):
        model, classes, values = self._create_model(sess, basename)

        predict_tensors = {}
        predict_tensors[model.lengths_key] = tf.compat.v1.saved_model.utils.build_tensor_info(model.lengths)

        for k, v in model.embeddings.items():
            try:
                predict_tensors[k] = tf.compat.v1.saved_model.utils.build_tensor_info(v.x)
            except:
                raise Exception('Unknown attribute in signature: {}'.format(v))

        sig_input = predict_tensors
        sig_output = SignatureOutput(classes, values)
        sig_name = 'tag_text'
        assets = create_assets(
            basename,
            sig_input, sig_output, sig_name,
            model.lengths_key,
            return_labels=self.return_labels,
            preproc=self.preproc_type()
        )
        return sig_input, sig_output, sig_name, assets


@export
@register_exporter(task='seq2seq', name='default')
class Seq2SeqTensorFlowExporter(TensorFlowExporter):
    """
    seq2seq has a source and tgt embedding layer.

    Please note that Tensorflow creates a state file where
    the target embeddings are not a dictionary. In order to reuse
    the embedding creation logic, I initialize a list of tuples
    as if items() was called on a dictionary. This also hides
    the correct key for target embeddings, so that is stored
    as a constant alongside the state keys.

    Perhaps this should be modified in the model, but this would decrease
    the readablity of the state file. It also doesn't make sense
    to generalize target embeddings to a dict since it's doubtful
    that we will ever target more than one embedding.

    :see baseline.python.tf.seq2seq.model:L411
    """
    SOURCE_STATE_EMBED_KEY = 'src_embeddings'
    TARGET_STATE_EMBED_KEY = 'tgt_embedding'
    TARGET_EMBED_KEY = 'tgt'

    def __init__(self, task, **kwargs):
        super(Seq2SeqTensorFlowExporter, self).__init__(task, **kwargs)
        self.return_labels = kwargs.get('return_labels', False)

    def _create_model(self, sess, basename, **kwargs):
        model = load_seq2seq_model(
            basename,
            sess=sess, predict=True,
            beam=self.task.config_params.get('beam', 30),
            **kwargs
        )
        return model, model.decoder.best, None

    def _create_rpc_call(self, sess, basename):
        model, classes, values = self._create_model(sess, basename)

        predict_tensors = {}
        predict_tensors[model.src_lengths_key] = tf.compat.v1.saved_model.utils.build_tensor_info(model.src_len)

        for k, v in model.src_embeddings.items():
            try:
                predict_tensors[k] = tf.compat.v1.saved_model.utils.build_tensor_info(v.x)
            except:
                raise Exception('Unknown attribute in signature: {}'.format(v))

        sig_input = predict_tensors
        sig_output = SignatureOutput(classes, values)
        sig_name = 'suggest_text'
        assets = create_assets(
            basename,
            sig_input, sig_output, sig_name,
            model.src_lengths_key,
            beam=model.decoder.beam_width,
            return_labels=self.return_labels,
            preproc=self.preproc_type(),
        )

        return sig_input, sig_output, sig_name, assets


def create_bundle(builder, output_path, basename, assets=None):
    """Creates the output files for an exported model.

    :builder the tensorflow saved_model builder.
    :output_path the path to export a model. this includes the model_version name.
    :assets a dictionary of assets to save alongside the model.
    """
    builder.save()

    model_name = os.path.basename(basename)
    directory = os.path.realpath(os.path.dirname(basename))
    save_to_bundle(output_path, directory, assets)


def create_assets(basename, sig_input, sig_output, sig_name, lengths_key=None, beam=None, return_labels=False, preproc='client'):
    """Save required variables for running an exported model from baseline's services.

    :basename the base model name. e.g. /path/to/tagger-26075
    :sig_input the input dictionary
    :sig_output the output namedTuple
    :lengths_key the lengths_key from the model.
        used to translate batch output. Exported models will return a flat list,
        and it needs to be reshaped into per-example lists. We use this key to tell
        us which feature holds the sequence lengths.

    """
    inputs = [k for k in sig_input]
    outputs =  sig_output._fields
    model_name = basename.split("/")[-1]
    directory = basename.split("/")[:-1]
    metadata = create_metadata(inputs, outputs, sig_name, model_name, lengths_key, beam=beam,
                               return_labels=return_labels, preproc=preproc)
    return metadata
