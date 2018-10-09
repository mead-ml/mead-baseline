import numpy as np
import tensorflow as tf
import json
import baseline
import os
from tensorflow.python.framework.errors_impl import NotFoundError
import mead.utils
import mead.exporters
from mead.tf.signatures import SignatureInput, SignatureOutput
from mead.tf.preprocessor import PreprocessorCreator
from baseline.utils import export
from baseline.tf.tfy import get_vocab_file_suffixes
FIELD_NAME = 'text/tokens'

__all__ = []

@export(__all__)
class TensorFlowExporter(mead.exporters.Exporter):
    DEFAULT_VOCABS = {"word", "char"}

    def __init__(self, task):
        super(TensorFlowExporter, self).__init__(task)

    def _run(self, sess, model_file, embeddings_set):
        pass

    def get_raw_post(self, tf_example):
        return tf_example[FIELD_NAME]

    def restore_model(self, sess, basename):
        saver = tf.train.Saver()
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess, basename)
        except NotFoundError:
            saver.restore(sess, basename + ".model")

    def run(self, model_file, embeddings, output_dir, model_version, use_preproc):
        embeddings_set = mead.utils.read_config_file(embeddings)
        embeddings_set = mead.utils.index_by_label(embeddings_set)
        with tf.Graph().as_default():
            config_proto = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config_proto) as sess:
                sig_input, sig_output, sig_name = self._run(sess, model_file, embeddings_set, use_preproc=use_preproc)
                
                output_path = os.path.join(tf.compat.as_bytes(output_dir),
                                   tf.compat.as_bytes(str(model_version)))
                
                print('Exporting trained model to %s' % output_path)
                builder = self._create_builder(sess, output_path, sig_input, sig_output, sig_name)
                builder.save()
                print('Successfully exported model to %s' % output_dir)

    @staticmethod
    def read_vocab(basename, ty):
        if ty is None:
            vocab_file = basename
            ty = 'word'
        else:
            vocab_file = "{}-{}.vocab".format(basename, ty)
        print('Reading {}'.format(vocab_file))
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)

        # Make a vocab list
        vocab_list = [''] * (len(vocab) + 1)

        for v, i in vocab.items():
            vocab_list[i] = v

        tok2index = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(vocab_list),
            default_value=0,
            dtype=tf.string,
            name='%s2index' % ty
        )
        return tok2index, vocab

    def load_labels(self, basename):

        label_file = '%s.labels' % basename
        with open(label_file, 'r') as f:
            labels = json.load(f)
        return labels

    def _create_example(self, extra_features_required):
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')

        feature_configs = {
            FIELD_NAME: tf.FixedLenFeature(shape=[], dtype=tf.string),
        }
        for other in extra_features_required:
            feature_configs[other] = tf.FixedLenFeature(shape=[], dtype=tf.string)

        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        return serialized_tf_example, tf_example

    def _create_vocabs(self, model_file):
        """
        :model_file the path-like object to the model and model name.
        :vocab_suffixes the list of vocab types. e.g. 'word', 'char', 'ner'.
        """
        vocabs = {}
        indices = {}
        if os.path.exists(model_file + '.vocab'):
            indices['word'], vocabs['word'] = TensorFlowExporter.read_vocab(model_file + '.vocab', ty=None)
        else:
            vocab_suffixes = get_vocab_file_suffixes(model_file)
            for suffix in vocab_suffixes:
                indices[suffix], vocabs[suffix] = TensorFlowExporter.read_vocab(model_file, suffix)

        return indices, vocabs

    def assign_char_lookup(self):
        upchars = tf.constant([chr(i) for i in range(65, 91)])
        self.lchars = tf.constant([chr(i) for i in range(97, 123)])
        self.upchars_lut = tf.contrib.lookup.index_table_from_tensor(mapping=upchars, num_oov_buckets=1, default_value=-1)

    def _initialize_embeddings_map(self, vocabs, embeddings_set):
        """
        generate a mapping of vocab_typ (word, char) to the embedding object.
        """
        embeddings = {}

        for vocab_type in vocabs.keys():
            dimension_size = self._get_embedding_dsz(embeddings_set, vocab_type)
            embeddings[vocab_type] = self._initialize_embedding(dimension_size, vocabs[vocab_type])

        return embeddings

    def _get_embedding_dsz(self, embeddings_set, embed_type):
        if embed_type == 'word':
            word_embeddings = self.task.config_params["word_embeddings"]
            return embeddings_set[word_embeddings["label"]]["dsz"]
        elif embed_type == 'char':
            return self.task.config_params["charsz"]
        else:
            extra_info = self.task.config_params["extended_embed_info"]

            if embed_type not in extra_info:
                raise ValueError("could not find embedding type in configuration. If \
the embedding is not of type 'word' or 'char', please fill in and put \
{ %s : {'dsz' : [ENTER_DIMENSION_SIZE_HERE] } } in the \
'extended_embed_info config object." % (embed_type))

            return extra_info[embed_type]['dsz']

    def _initialize_embedding(self, dimensions_size, vocab):
        return baseline.RandomInitVecModel(dimensions_size, vocab, False)

    def _run_preproc(self, model_params, vocabs, model_file, indices, extra_features_required):
        serialized_tf_example, tf_example = self._create_example(extra_features_required)
        raw_posts = tf_example[FIELD_NAME]
        
        preprocessed, lengths = self._create_preprocessed_input(tf_example, 
                                                            model_file, 
                                                            indices,
                                                            extra_features_required)
    
        model_params["x"] = preprocessed['word']
        if 'char' in vocabs:
            model_params['xch'] = preprocessed['char']
        for other in extra_features_required:
            model_params[other] = preprocessed[other]

        return serialized_tf_example, tf_example, raw_posts, lengths

    def _create_preprocessed_input(self, tf_example, 
                                   model_file, 
                                   indices, 
                                   extra_features_required):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        mxlen, mxwlen = self._get_max_lens(model_file)

        preprocessor = PreprocessorCreator(
            indices, self.lchars, self.upchars_lut,
            self.task, FIELD_NAME, extra_features_required, mxlen, mxwlen
        )
        types = {k: tf.int64 for k in indices}
        preprocessed, lengths = tf.map_fn(
            preprocessor.preproc_post, tf_example,
            dtype=(types, tf.int32), back_prop=False
        )

        return preprocessed, lengths

    def _get_max_lens(self, base_name):
        mxlen = self.task.config_params['preproc']['mxlen']
        mxwlen = self.task.config_params['preproc'].get('mxwlen')
        state = baseline.utils.read_json("{}.state".format(base_name))
        if 'mxlen' in state:
            mxlen = state['mxlen']
        # What should be called mxwlen is called maxw in the state object of this is for backwards compatibility.
        if 'maxw' in state:
            mxwlen = state['maxw']
        if 'mxwlen' in state:
            mxwlen = state['mxwlen']
        return mxlen, mxwlen

    def _create_builder(self, sess, output_path, sig_input, sig_output, sig_name):
        """
        create the SavedModelBuilder with standard endpoints.

        we reuse the classify constants from tensorflow to define the predict
        endpoint so that we can call the output by classes/scores.
        """
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)

        classes_output_tensor = tf.saved_model.utils.build_tensor_info(
            sig_output.classes)
        
        output_def_map = {
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                        classes_output_tensor
        }
        if sig_output.scores is not None:
            scores_output_tensor = tf.saved_model.utils.build_tensor_info(sig_output.scores)
            output_def_map[tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES] = scores_output_tensor

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=sig_input.predict,
                outputs=output_def_map,  # we reuse classify constants here.
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        definition = {}
        definition[sig_name] = prediction_signature
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=definition,
            legacy_init_op=legacy_init_op)

        return builder


@export(__all__)
class ClassifyTensorFlowExporter(TensorFlowExporter):

    def __init__(self, task):
        super(ClassifyTensorFlowExporter, self).__init__(task)

    def _run(self, sess, model_file, embeddings_set, use_preproc=True):
        indices, vocabs = self._create_vocabs(model_file)
        extra_features_required = [x for x in vocabs.keys() if x not in TensorFlowExporter.DEFAULT_VOCABS]

        self.assign_char_lookup()

        labels = self.load_labels(model_file)
        mxlen, mxwlen = self._get_max_lens(model_file)

        model_params = self.task.config_params["model"]

        if use_preproc:
            serialized_tf_example, tf_example, raw_posts, lengths = self._run_preproc(model_params, 
                                                                            vocabs, 
                                                                            model_file, 
                                                                            indices, 
                                                                            extra_features_required)
        
        model_params["pkeep"] = 1
        model_params["sess"] = sess
        model_params["maxs"] = mxlen
        model_params["maxw"] = mxwlen
        print(model_params)

        embeddings = self._initialize_embeddings_map(vocabs, embeddings_set)
        model = baseline.tf.classify.create_model(embeddings, labels, **model_params)
        softmax_output = tf.nn.softmax(model.logits)

        values, indices = tf.nn.top_k(softmax_output, len(labels))
        class_tensor = tf.constant(model.labels)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))
        self.restore_model(sess, model_file)
        
        if use_preproc:
            sig_input = SignatureInput(serialized_tf_example, tf_example, extra_features_required)
        else:
            sig_input = SignatureInput(None, None, extra_features_required, model=model)

        sig_output = SignatureOutput(classes, values)

        return sig_input, sig_output, 'predict_text'


@export(__all__)
class TaggerTensorFlowExporter(TensorFlowExporter):

    def __init__(self, task):
        super(TaggerTensorFlowExporter, self).__init__(task)


    def _create_model(self, vocabs, labels, embeddings_set, mxlen, model_params):
        embeddings = self._initialize_embeddings_map(vocabs, embeddings_set)
        model = baseline.tf.tagger.create_model(labels, embeddings, **model_params)
        model.create_loss()

        softmax_output = tf.nn.softmax(model.probs)
        values, indices = tf.nn.top_k(softmax_output, 1)

        start_np = np.full((1, 1, len(labels)), -1e4, dtype=np.float32)
        start_np[:, 0, labels['<GO>']] = 0
        start = tf.constant(start_np)
        model.probs = tf.concat([start, model.probs], 1)

        if model.crf is True:
            indices, _ = tf.contrib.crf.crf_decode(model.probs, model.A, tf.constant([mxlen + 1]))## We are assuming the batchsz is 1 here
            indices = indices[:, 1:]

        list_of_labels = [''] * len(labels)
        for label, idval in labels.items():
            list_of_labels[idval] = label

        class_tensor = tf.constant(list_of_labels)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))

        return classes, values, model

    def _run(self, sess, model_file, embeddings_set, use_preproc=True):
        mxlen, mxwlen = self._get_max_lens(model_file)
        indices, vocabs = self._create_vocabs(model_file)
        self.assign_char_lookup()

        labels = self.load_labels(model_file)

        extra_features_required = [x for x in vocabs.keys() if x not in TensorFlowExporter.DEFAULT_VOCABS]
        model_params = self.task.config_params["model"]

        lengths = []

        if use_preproc:
            serialized_tf_example, tf_example, raw_posts, lengths = self._run_preproc(model_params, 
                                                                            vocabs, 
                                                                            model_file, 
                                                                            indices, 
                                                                            extra_features_required)
            model_params["lengths"] = lengths


        model_params["pkeep"] = 1
        model_params["sess"] = sess
        model_params["maxs"] = mxlen
        model_params["maxw"] = mxwlen
        model_params['span_type'] = self.task.config_params['train'].get('span_type')
        print(model_params)

        classes, values, model = self._create_model(vocabs, 
                                                    labels, 
                                                    embeddings_set, 
                                                    mxlen, 
                                                    model_params)
        self.restore_model(sess, model_file)
        
        if use_preproc:
            sig_input = SignatureInput(serialized_tf_example, tf_example, extra_features_required)
        else:
            sig_input = SignatureInput(None, None, extra_features_required + ['lengths'], model=model)

        sig_output = SignatureOutput(classes, values)

        return sig_input, sig_output, 'tag_text'


@export(__all__)
class Seq2SeqTensorFlowExporter(TensorFlowExporter):

    def __init__(self, task):
        super(Seq2SeqTensorFlowExporter, self).__init__(task)

    @staticmethod
    def read_input_vocab(basename):
        vocab_file = '%s-1.vocab' % basename
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)

        # Make a vocab list
        vocab_list = [''] * len(vocab)

        for v, i in vocab.items():
            vocab_list[i] = v

        word2input = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(vocab_list),
            default_value=0,
            dtype=tf.string,
            name='word2input'
        )
        return word2input, vocab

    @staticmethod
    def read_output_vocab(basename):
        vocab_file = '%s-2.vocab' % basename
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)

        # Make a vocab list
        vocab_list = [''] * len(vocab)

        for v, i in vocab.items():
            vocab_list[i] = v

        output2word = tf.contrib.lookup.index_to_string_table_from_tensor(
            tf.constant(vocab_list),
            default_value='<PAD>',
            name='output2word'
        )

        return output2word, vocab

    def get_dsz(self, embeddings_set):
        embeddings_section = self.task.config_params['word_embeddings']
        if embeddings_section.get('label', None) is not None:
            embed_label = embeddings_section['label']
            dsz = embeddings_set[embed_label]['dsz']
        else:
            dsz = embeddings_section['dsz']
        return dsz

    def _preproc_post_creator(self):
        word2input = self.word2input

        def preproc_post(raw_post):
            # raw_post is a "scalar string tensor"
            # (https://www.tensorflow.org/versions/r0.12/api_docs/python/image/encoding_and_decoding)
            # Split the input string, assuming that whitespace is splitter
            # The client should perform any required tokenization for us and join on ' '
            #raw_post = tf.Print(raw_post, [raw_post])
            mxlen = self.task.config_params['preproc']['mxlen']
            raw_tokens = tf.string_split(tf.reshape(raw_post, [-1])).values
            npost = tf.reduce_join(raw_tokens[:mxlen], separator=" ")
            tokens = tf.string_split(tf.reshape(npost, [-1]))
            sentence_length = tf.size(tokens)

            # Convert the string values to word indices (ints)
            indices = word2input.lookup(tokens)

            # Reshape them out to the proper length
            reshaped = tf.sparse_reshape(indices, shape=[-1])
            reshaped = tf.sparse_reset_shape(reshaped, new_shape=[mxlen])

            # Now convert to a dense representation
            dense = tf.sparse_tensor_to_dense(reshaped)
            dense = tf.contrib.framework.with_shape([mxlen], dense)
            dense = tf.cast(dense, tf.int32)
            return dense, sentence_length
        return preproc_post

    def _run(self, sess, model_file, embeddings_set):

        self.word2input, vocab1 = Seq2SeqTensorFlowExporter.read_input_vocab(model_file)
        self.output2word, vocab2 = Seq2SeqTensorFlowExporter.read_output_vocab(model_file)

        # Make the TF example, network input
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            FIELD_NAME: tf.FixedLenFeature(shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        raw_posts = tf_example[FIELD_NAME]

        # Run for each post
        dense, length = tf.map_fn(self._preproc_post_creator(), raw_posts,
                                  dtype=(tf.int32, tf.int32))

        model_params = self.task.config_params["model"]
        model_params["dsz"] = self.get_dsz(embeddings_set)
        model_params["src"] = dense
        model_params["src_len"] = length
        model_params["mx_tgt_len"] = self.task.config_params["preproc"]["mxlen"]
        model_params["tgt_len"] = 1
        model_params["pkeep"] = 1
        model_params["sess"] = sess
        model_params["predict"] = True
        print(model_params)
        model = baseline.tf.seq2seq.create_model(vocab1, vocab2, **model_params)
        output = self.output2word.lookup(tf.cast(model.best, dtype=tf.int64))

        self.restore_model(sess, model_file)

        sig_input = SignatureInput(serialized_tf_example, raw_posts)
        sig_output = SignatureOutput(classes, None)

        return sig_input, sig_output, 'suggest_text'
