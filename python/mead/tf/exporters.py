import tensorflow as tf
import json
import baseline
import os
import mead.utils
import mead.exporters
from baseline.utils import export

FIELD_NAME = 'text/tokens'

__all__ = []

@export(__all__)
class TensorFlowExporter(mead.exporters.Exporter):

    def __init__(self, task):
        super(TensorFlowExporter, self).__init__(task)
        
    def _run(self, sess, model_file, embeddings_set):
        pass

    def restore_model(self, sess, basename):
        saver = tf.train.Saver()
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, basename + '.model')

    def run(self, model_file, embeddings, output_dir, model_version, **kwargs):
        embeddings_set = mead.utils.index_by_label(embeddings)
        with tf.Graph().as_default():
            config_proto = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config_proto) as sess:
                self._run(sess, model_file, embeddings_set, output_dir, model_version)

    def load_labels(self, basename):

        label_file = '%s.labels' % basename
        with open(label_file, 'r') as f:
            labels = json.load(f)
        return labels

    
@export(__all__)
class ClassifyTensorFlowExporter(TensorFlowExporter):

    def __init__(self, task):
        super(ClassifyTensorFlowExporter, self).__init__(task)

    @staticmethod
    def read_vocab(basename):
        vocab_file = '%s.vocab' % basename
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)

        # Make a vocab list
        vocab_list = [''] * len(vocab)

        for v, i in vocab.items():
            vocab_list[i] = v

        word2index = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(vocab_list),
            default_value=0,
            dtype=tf.string,
            name='word2index'
        )
        return word2index, vocab

    def _preproc_post_creator(self):
        word2index = self.word2index

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
            # Convert the string values to word indices (ints)
            indices = word2index.lookup(tokens)

            # Reshape them out to the proper length
            reshaped = tf.sparse_reshape(indices, shape=[-1])
            reshaped = tf.sparse_reset_shape(reshaped, new_shape=[mxlen])

            # Now convert to a dense representation
            dense = tf.sparse_tensor_to_dense(reshaped)
            return dense
        return preproc_post

    def _run(self, sess, model_file, embeddings_set, output_dir, model_version):
        self.word2index, vocab = ClassifyTensorFlowExporter.read_vocab(model_file)
        labels = self.load_labels(model_file)
        # Make the TF example, network input
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            FIELD_NAME: tf.FixedLenFeature(shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        raw_posts = tf_example[FIELD_NAME]

        dense = tf.map_fn(self._preproc_post_creator(), raw_posts, dtype=tf.int64)
        word_embeddings = self.task.config_params["word_embeddings"]
        dsz = embeddings_set[word_embeddings["label"]]["dsz"]
        init_vectors = baseline.RandomInitVecModel(dsz, vocab, False)
        print(len(init_vectors.weights), len(vocab), init_vectors.vsz)
        model_params = self.task.config_params["model"]
        model_params["x"] = dense
        model_params["pkeep"] = 1
        model_params["sess"] = sess
        print(model_params)
        model = baseline.tf.classify.create_model({'word': init_vectors}, labels, **model_params)
        softmax_output = tf.nn.softmax(model.logits)

        values, indices = tf.nn.top_k(softmax_output, len(labels))
        class_tensor = tf.constant(model.labels)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))
        self.restore_model(sess, model_file)
        output_path = os.path.join(tf.compat.as_bytes(output_dir),
                                   tf.compat.as_bytes(str(model_version)))

        print('Exporting trained model to %s' % output_path)
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)

        # Build the signature_def_map.
        classify_inputs_tensor = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
        classes_output_tensor = tf.saved_model.utils.build_tensor_info(
            classes)
        scores_output_tensor = tf.saved_model.utils.build_tensor_info(values)

        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                        classify_inputs_tensor
                },
                outputs={
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                        classes_output_tensor,
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                        scores_output_tensor
                },
                method_name=tf.saved_model.signature_constants.
                    CLASSIFY_METHOD_NAME)
        )

        predict_inputs_tensor = tf.saved_model.utils.build_tensor_info(raw_posts)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'tokens': predict_inputs_tensor},
                outputs={
                    'classes': classes_output_tensor,
                    'scores': scores_output_tensor
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_text':
                    prediction_signature,
                tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    classification_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
        print('Successfully exported model to %s' % output_dir)


@export(__all__)
class TaggerTensorFlowExporter(TensorFlowExporter):

    def __init__(self, task):
        super(TaggerTensorFlowExporter, self).__init__(task)

    @staticmethod
    def _find_files_by_type(model_file, filetype):
        """
        I need to find all vocab files within a tensorflow model directory.
        To do this, we will grab the directory and do a lookup by filetype.

        we rely on the fact that vocab files end in .vocab.

        :returns the file names without the filetype.
        """
        matching_files = []

        filetype_ending = "." + filetype
        basepath = TaggerTensorFlowExporter._get_dir_from_model_file(model_file)
        if not os.path.isdir(basepath):
            raise IOError("not a model directory")

        for filename in os.listdir(basepath):
            if filename.endswith(filetype_ending):
                filename_without_ending = filename[:-len(filetype_ending)]
                matching_files.append(filename_without_ending)

        return matching_files

    @staticmethod
    def _get_vocab_file_suffixes(model_file, filenames):
        """
        because our operations assume knowledge of the model name, we 
        only need toreturn the suffix appended onto the end of the model 
        name in the file.

        we make the assumption that a suffix is denoted with a hyphen.

        e.g.  a vocab file name = tagger-model-tf-30803-word.vocab
              would return ['word']

        model_file: the nonspecific path to the model. this could be 
                    /data/model/<model_name>. we need to remove the model name.
        """
        model_name = model_file.split('/')[-1]
        # the length of the name plus 1 for the hyphen separating the suffix.
        return [x[len(model_name)+1:] for x in filenames]

    @staticmethod
    def _get_dir_from_model_file(model_file):
        """
        model file is a path, but to a nonspecific model name.

        this method grabs the directory.
        """
        return os.path.dirname(model_file)

    @staticmethod
    def read_vocab(basename, ty):
        vocab_file = '%s-%s.vocab' % (basename, ty)
        print('Reading {}', vocab_file)
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

    def _preproc_post_creator(self):
        """
        indices are created during vocab creation.
        """
        word2index = self.word2index
        char2index = self.char2index
        lchars = self.lchars
        upchars_lut = self.upchars_lut
        task = self.task
        def preproc_post(raw_post):
            # Split the input string, assuming that whitespace is splitter
            # The client should perform any required tokenization for us and join on ' '

            # WARNING: This can be a bug if the user defaults the values (-1)
            # for conll, the mxlen=124, for idr, the mxlen is forced to a max BPTT
            # for twpos, the mxlen=38
            # this should probably be fixed by serializing the mxlen of the model
            # or rereading it from the tensor from file
            mxlen = task.config_params['preproc']['mxlen']
            mxwlen = task.config_params['preproc']['mxwlen']

            #raw_post = tf.Print(raw_post, [raw_post])
            raw_tokens = tf.string_split(tf.reshape(raw_post, [-1])).values
            # sentence length <= mxlen
            nraw_post = tf.reduce_join(raw_tokens[:mxlen], separator=" ")
            # vocab has only lowercase words
            split_chars = tf.string_split(tf.reshape(nraw_post, [-1]), delimiter="").values
            upchar_inds = upchars_lut.lookup(split_chars)
            lc_raw_post = tf.reduce_join(tf.map_fn(lambda x: tf.cond(x[0] > 25,
                                                                     lambda: x[1],
                                                                     lambda: lchars[x[0]]),
                                                   (upchar_inds, split_chars), dtype=tf.string))
            word_tokens = tf.string_split(tf.reshape(lc_raw_post, [-1]))

            # numchars per word should be <= mxwlen
            unchanged_word_tokens = tf.string_split(tf.reshape(nraw_post, [-1]))
            culled_word_token_vals = tf.substr(unchanged_word_tokens.values, 0, mxwlen)
            char_tokens = tf.string_split(culled_word_token_vals, delimiter='')
            word_indices = word2index.lookup(word_tokens)
            char_indices = char2index.lookup(char_tokens)

            # Reshape them out to the proper length
            reshaped_words = tf.sparse_reshape(word_indices, shape=[-1])
            sentence_length = tf.size(reshaped_words)  # tf.shape if 2 dims needed

            reshaped_words = tf.sparse_reset_shape(reshaped_words, new_shape=[mxlen])
            reshaped_chars = tf.sparse_reset_shape(char_indices, new_shape=[mxlen, mxwlen])

            # Now convert to a dense representation
            x = tf.sparse_tensor_to_dense(reshaped_words)
            x = tf.contrib.framework.with_shape([mxlen], x)
            xch = tf.sparse_tensor_to_dense(reshaped_chars)
            xch = tf.contrib.framework.with_shape([mxlen, mxwlen], xch)
            return x, xch, sentence_length
        return preproc_post

    def restore_model(self, sess, basename):
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, basename)

    def _run(self, sess, model_file, embeddings_set, output_dir, model_version):
        vocab_files = TaggerTensorFlowExporter._find_files_by_type(model_file, 'vocab')
        vocab_suffixes = TaggerTensorFlowExporter._get_vocab_file_suffixes(model_file, vocab_files)
        indices, vocabs = self._create_vocabs(model_file, vocab_suffixes)

        self.assign_char_lookup()

        labels = self.load_labels(model_file)
        # Make the TF example, network input
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            FIELD_NAME: tf.FixedLenFeature(shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        raw_posts = tf_example[FIELD_NAME]

        # self._preproc_post_creator requires these indices.
        self.word2index = indices['word']
        self.char2index = indices['char']
        # Run for each post
        x, xch, lengths = tf.map_fn(self._preproc_post_creator(), raw_posts,
                                    dtype=(tf.int64, tf.int64, tf.int32),
                                    back_prop=False)

        embeddings = self._initialize_embeddings_map(vocabs, embeddings_set)

        # WARNING: This can be a bug if the user defaults the values (-1)
        # for conll, the mxlen=124, for idr, the mxlen is forced to a max BPTT
        # for twpos, the mxlen=38
        # this should probably be fixed by serializing the mxlen of the model
        # or rereading it from the tensor from file
        mxlen = self.task.config_params['preproc']['mxlen']
        mxwlen = self.task.config_params['preproc']['mxwlen']

        model_params = self.task.config_params["model"]
        model_params["x"] = x
        model_params["xch"] = xch
        model_params["lengths"] = lengths
        model_params["pkeep"] = 1
        model_params["sess"] = sess
        model_params["maxs"] = mxlen
        model_params["maxw"] = mxwlen
        print(model_params)
        model = baseline.tf.tagger.create_model(labels, embeddings, **model_params)
        model.create_loss()

        softmax_output = tf.nn.softmax(model.probs)
        values, indices = tf.nn.top_k(softmax_output, 1)

        if model.crf is True:
            indices, _ = tf.contrib.crf.crf_decode(model.probs, model.A, tf.constant([mxlen ]))## We are assuming the batchsz is 1 here

        list_of_labels = [''] * len(labels)
        for label, idval in labels.items():
            list_of_labels[idval] = label

        class_tensor = tf.constant(list_of_labels)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))
        self.restore_model(sess, model_file)
        output_path = os.path.join(tf.compat.as_bytes(output_dir),
                                   tf.compat.as_bytes(str(model_version)))

        print('Exporting trained model to %s' % output_path)
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)

        # Build the signature_def_map.
        classify_inputs_tensor = tf.saved_model.utils.build_tensor_info(
            serialized_tf_example)
        classes_output_tensor = tf.saved_model.utils.build_tensor_info(
            classes)
        scores_output_tensor = tf.saved_model.utils.build_tensor_info(values)

        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                        classify_inputs_tensor
                },
                outputs={
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                        classes_output_tensor,
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                        scores_output_tensor
                },
                method_name=tf.saved_model.signature_constants.
                    CLASSIFY_METHOD_NAME)
        )

        predict_inputs_tensor = tf.saved_model.utils.build_tensor_info(raw_posts)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'tokens': predict_inputs_tensor},
                outputs={
                    'classes': classes_output_tensor,
                    'scores': scores_output_tensor
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'tag_text':
                    prediction_signature,
                tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    classification_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
        print('Successfully exported model to %s' % output_dir)

    def _create_vocabs(self, model_file, vocab_suffixes):
        """
        :model_file the path-like object to the model and model name.
        :vocab_suffixes the list of vocab types. e.g. 'word', 'char', 'ner'.
        """
        vocabs = {}
        indices = {}
        for suffix in vocab_suffixes:
            index, vocab = TaggerTensorFlowExporter.read_vocab(model_file, suffix)
            vocabs[suffix] = vocab
            indices[suffix] = index

        return indices, vocabs

    def assign_char_lookup(self):
        upchars = tf.constant([chr(i) for i in range(65, 91)])
        self.lchars = tf.constant([chr(i) for i in range(97, 123)])
        self.upchars_lut = tf.contrib.lookup.index_table_from_tensor(mapping=upchars, num_oov_buckets=1, default_value=-1)
        
    def _initialize_embeddings_map(self, vocabs, embeddings_set):
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

    def _run(self, sess, model_file, embeddings_set, output_dir, model_version):

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
        output_path = os.path.join(tf.compat.as_bytes(output_dir),
                                   tf.compat.as_bytes(str(model_version)))

        print('Exporting trained model to %s' % output_path)
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)
        # Build the signature_def_map.
        classify_inputs_tensor = tf.saved_model.utils.build_tensor_info(
            serialized_tf_example)
        classes_output_tensor = tf.saved_model.utils.build_tensor_info(
            output)
        #scores_output_tensor = tf.saved_model.utils.build_tensor_info(values)

        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                        classify_inputs_tensor
                },
                outputs={
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                        classes_output_tensor #,
                    #tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                    #scores_output_tensor
                },
                method_name=tf.saved_model.signature_constants.
                    CLASSIFY_METHOD_NAME)
        )

        predict_inputs_tensor = tf.saved_model.utils.build_tensor_info(raw_posts)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'tokens': predict_inputs_tensor},
                outputs={
                    'classes': classes_output_tensor #,
                    #'scores': scores_output_tensor
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'suggest_text':
                    prediction_signature,
                tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    classification_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
        print('Successfully exported model to %s' % output_dir)
