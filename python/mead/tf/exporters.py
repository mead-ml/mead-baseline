import baseline
import os
from tensorflow.python.framework.errors_impl import NotFoundError
import mead.exporters
from mead.exporters import register_exporter
from baseline.tf.embeddings import *
from baseline.utils import export, read_json, ls_props, Offsets
from collections import namedtuple

FIELD_NAME = 'text/tokens'

__all__ = []
exporter = export(__all__)


SignatureOutput = namedtuple("SignatureOutput", ("classes", "scores"))


@exporter
class TensorFlowExporter(mead.exporters.Exporter):

    def __init__(self, task):
        super(TensorFlowExporter, self).__init__(task)

    def _run(self, sess, basename):
        pass

    def _restore_checkpoint(self, sess, basename):
        saver = tf.train.Saver()
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess, basename)
        except NotFoundError:
            saver.restore(sess, basename + ".model")

    def run(self, basename, output_dir, model_version, **kwargs):
        """
        :param embeddings_set an object of all embeddings. read from the embeddings json config.
        :param feature_descs an object describing the features. typically each key will be a
            dict with keys (type, dsz, vsz)
        """
        with tf.Graph().as_default():
            config_proto = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config_proto) as sess:
                sig_input, sig_output, sig_name = self._create_rpc_call(sess, basename)
                output_path = os.path.join(tf.compat.as_bytes(output_dir), tf.compat.as_bytes(str(model_version)))
                print('Exporting trained model to %s' % output_path)
                builder = self._create_saved_model_builder(sess, output_path, sig_input, sig_output, sig_name)
                builder.save()
                print('Successfully exported model to %s' % output_dir)

    def _create_saved_model_builder(self, sess, output_path, sig_input, sig_output, sig_name):
        """
        create the SavedModelBuilder with standard endpoints.

        we reuse the classify constants from tensorflow to define the predict
        endpoint so that we can call the output by classes/scores.
        """
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)

        classes_output_tensor = tf.saved_model.utils.build_tensor_info(
            sig_output.classes)
        
        output_def_map = {
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classes_output_tensor
        }
        if sig_output.scores is not None:
            scores_output_tensor = tf.saved_model.utils.build_tensor_info(sig_output.scores)
            output_def_map[tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES] = scores_output_tensor

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=sig_input,
                outputs=output_def_map,  # we reuse classify constants here.
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        definition = dict({})
        definition[sig_name] = prediction_signature

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=definition,
            legacy_init_op=legacy_init_op)

        return builder

    def _create_rpc_call(self, sess, basename):
        pass


@exporter
@register_exporter(task='classify', name='default')
class ClassifyTensorFlowExporter(TensorFlowExporter):

    def __init__(self, task):
        super(ClassifyTensorFlowExporter, self).__init__(task)

    def _create_model(self, sess, basename):
        # Read the labels
        labels = read_json(basename + '.labels')

        # Get the parameters from MEAD
        model_params = self.task.config_params["model"]
        model_params["sess"] = sess

        # Read the state file
        state = read_json(basename + '.state')

        # Re-create the embeddings sub-graph
        embeddings = dict()
        for key, class_name in state['embeddings'].items():
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            Constructor = eval(class_name)
            embeddings[key] = Constructor(key, **embed_args)

        # Instantiate a graph
        model = baseline.model.create_model_for(self.task.task_name(), embeddings, labels, **model_params)

        # Set the properties of the model from the state file
        for prop in ls_props(model):
            if prop in state:
                setattr(model, prop, state[prop])

        # Append to the graph for class output
        values, indices = tf.nn.top_k(model.probs, len(labels))
        class_tensor = tf.constant(model.labels)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))

        # Restore the checkpoint
        self._restore_checkpoint(sess, basename)
        return model, classes, values

    def _create_rpc_call(self, sess, basename):
        model, classes, values = self._create_model(sess, basename)

        predict_tensors = {}

        for k, v in model.embeddings.items():
            try:
                predict_tensors[k] = tf.saved_model.utils.build_tensor_info(v.x)
            except:
                raise Exception('Unknown attribute in signature: {}'.format(v))

        sig_input = predict_tensors
        sig_output = SignatureOutput(classes, values)
        return sig_input, sig_output, 'predict_text'


@exporter
@register_exporter(task='tagger', name='default')
class TaggerTensorFlowExporter(TensorFlowExporter):

    def __init__(self, task):
        super(TaggerTensorFlowExporter, self).__init__(task)

    def _create_model(self, sess, basename):
        labels = read_json(basename + '.labels')
        model_params = self.task.config_params["model"]
        model_params["sess"] = sess

        state = read_json(basename + '.state')
        model_params['span_type'] = state['span_type']
        # Re-create the embeddings sub-graph
        embeddings = dict()
        for key, class_name in state['embeddings'].items():
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            Constructor = eval(class_name)
            embeddings[key] = Constructor(key, **embed_args)
                
        model = baseline.model.create_model_for(self.task.task_name(), embeddings, labels, **model_params)

        for prop in ls_props(model):
            if prop in state:
                setattr(model, prop, state[prop])

        model.create_loss()

        softmax_output = tf.nn.softmax(model.probs)
        values, indices = tf.nn.top_k(softmax_output, 1)

        start_np = np.full((1, 1, len(labels)), -1e4, dtype=np.float32)
        start_np[:, 0, Offsets.GO] = 0
        start = tf.constant(start_np)
        model.probs = tf.concat([start, model.probs], 1)

        ones = tf.fill(tf.shape(model.lengths), 1)
        lengths = tf.add(model.lengths, ones)

        if model.crf is True:
            indices, _ = tf.contrib.crf.crf_decode(model.probs, model.A, lengths)
            indices = indices[:, 1:]

        # list_of_labels = [''] * len(labels)
        # for label, idval in labels.items():
        #     list_of_labels[idval] = label

        # class_tensor = tf.constant(list_of_labels)
        # table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        # classes = table.lookup(tf.to_int64(indices))
        self._restore_checkpoint(sess, basename)

        return model, indices, values

    def _create_rpc_call(self, sess, basename):
        model, classes, values = self._create_model(sess, basename)

        predict_tensors = {}
        predict_tensors[model.lengths_key] = tf.saved_model.utils.build_tensor_info(model.lengths)

        for k, v in model.embeddings.items():
            try:
                predict_tensors[k] = tf.saved_model.utils.build_tensor_info(v.x)
            except:
                raise Exception('Unknown attribute in signature: {}'.format(v))

        sig_input = predict_tensors
        print(sig_input.keys())
        sig_output = SignatureOutput(classes, values)
        return sig_input, sig_output, 'tag_text'


@exporter
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

    def __init__(self, task):
        super(Seq2SeqTensorFlowExporter, self).__init__(task)

    def init_embeddings(self, embeddings_map, basename):
        embeddings = dict()
        for key, class_name in embeddings_map:
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            Constructor = eval(class_name)
            embeddings[key] = Constructor(key, **embed_args)
        
        return embeddings

    def _create_model(self, sess, basename):
        model_params = self.task.config_params["model"]
        model_params["sess"] = sess
        model_params['predict'] = True
        model_params['beam'] = self.task.config_params.get('beam', 30)

        state = read_json(basename + '.state')
        if not state:
            raise RuntimeError("state file not found or is empty")

        model_params["src_lengths_key"] = state["src_lengths_key"]
        self.length_key = state["src_lengths_key"]

        # Re-create the embeddings sub-graph
        embeddings = self.init_embeddings(state[self.SOURCE_STATE_EMBED_KEY].items(), basename)

        # create the taget embeddings. there's only one.
        target_embeddings = self.init_embeddings([
            (self.TARGET_EMBED_KEY, state[self.TARGET_STATE_EMBED_KEY])
        ], basename)
        target_embeddings = target_embeddings[self.TARGET_EMBED_KEY]

        model = baseline.model.create_model_for(self.task.task_name(), embeddings, target_embeddings, **model_params)

        for prop in ls_props(model):
            if prop in state:
                setattr(model, prop, state[prop])

        # classes = model.tgt_embedding.lookup(tf.cast(model.best, dtype=tf.int64))
        classes = model.best
        self._restore_checkpoint(sess, basename)

        return model, classes, None

    def _create_rpc_call(self, sess, basename):
        model, classes, values = self._create_model(sess, basename)

        predict_tensors = {}
        predict_tensors[self.length_key] = tf.saved_model.utils.build_tensor_info(model.src_len)

        for k, v in model.src_embeddings.items():
            try:
                predict_tensors[k] = tf.saved_model.utils.build_tensor_info(v.x)
            except:
                raise Exception('Unknown attribute in signature: {}'.format(v))

        sig_input = predict_tensors
        print(sig_input)
        sig_output = SignatureOutput(classes, values)
        return sig_input, sig_output, 'suggest_text'
        