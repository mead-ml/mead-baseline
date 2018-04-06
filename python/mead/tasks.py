import baseline
import json
import numpy as np
import logging
import logging.config
import mead.utils


class Task(object):
    TASK_REGISTRY = {}

    def __init__(self, logger_file):
        super(Task, self).__init__()
        self.config_params = None
        self.ExporterType = None
        self._configure_logger(logger_file)

    def _configure_logger(self, logger_file):
        with open(logger_file) as f:
            config = json.load(f)
            logging.config.dictConfig(config)

    @staticmethod
    def get_task_specific(task, logging_config):
        config = Task.TASK_REGISTRY[task](logging_config)
        return config

    def read_config(self, config_file, datasets_index):
        """
        Read the config file and the datasets index

        Between the config file and the dataset index, we have enough information
        to configure the backend and the models.  We can also initialize the data readers

        :param config_file: The config file
        :param datasets_index: The index of datasets
        :return:
        """
        datasets_set = mead.utils.index_by_label(datasets_index)
        self.config_params = self._read_config(config_file)
        self._setup_task()
        self._configure_reporting()
        self.dataset = datasets_set[self.config_params['dataset']]
        self.reader = self._create_task_specific_reader()

    def initialize(self, embeddings_index):
        """
        Load the vocabulary using the readers and then load any embeddings required

        :param embeddings_index: The index of embeddings
        :return:
        """
        pass

    def _create_task_specific_reader(self):
        """
        Create a task specific reader, based on the config
        :return:
        """
        pass

    def _setup_task(self):
        """
        This (pure) method provides the task-specific setup
        :return:
        """
        pass

    def _load_dataset(self):
        pass

    def _create_model(self):
        pass

    def train(self):
        """
        Do training
        :return:
        """
        self._load_dataset()
        model = self._create_model()
        self.task.fit(model, self.train_data, self.valid_data, self.test_data, **self.config_params['train'])
        return model

    def _configure_reporting(self):
        reporting = {
            "logging": True,
            "visdom": self.config_params.get('visdom', False),
            "tensorboard": self.config_params.get('tensorboard', False)
        }
        reporting = baseline.setup_reporting(**reporting)
        self.config_params['train']['reporting'] = reporting
        logging.basicConfig(level=logging.DEBUG)

    @staticmethod
    def _create_embeddings_from_file(embed, vocab, unif, keep_unused):
        EmbeddingT = baseline.GloVeModel if embed.endswith('.txt') else baseline.Word2VecModel
        return EmbeddingT(embed, vocab, unif_weight=unif, keep_unused=keep_unused)

    def _create_embeddings(self, embeddings_set, vocabs):

        unif = self.config_params['unif']
        keep_unused = self.config_params.get('keep_unused', False)

        if 'word' in vocabs:
            embeddings_section = self.config_params['word_embeddings']
            embed_label = embeddings_section.get('label', None)

            embeddings = dict()
            if embed_label is not None:
                embed_file = embeddings_set[embed_label]['file']

                embeddings['word'] = Task._create_embeddings_from_file(embed_file, vocabs['word'], unif=unif, keep_unused=keep_unused)
            else:
                dsz = embeddings_section['dsz']
                embeddings['word'] = baseline.RandomInitVecModel(dsz, vocabs['word'], unif_weight=unif)

        if 'char' in vocabs:
            if self.config_params.get('charsz', -1) > 0:
                embeddings['char'] = baseline.RandomInitVecModel(self.config_params['charsz'], vocabs['char'], unif_weight=unif)

        extended_embed_info = self.config_params.get('extended_embed_info', {}),
        for key, vocab in vocabs.items():
            if key in extended_embed_info:
                print('Adding extended feature embeddings {}'.format(key))
                ext_embed = None if extended_embed_info[key].get("embedding", None) is None \
                    else extended_embed_info[key]["embedding"]
                ext_emb_dsz = extended_embed_info[key].get("dsz", None)
                if ext_embed is not None:
                    EmbeddingT = baseline.GloVeModel if ext_embed.endswith('.txt') else baseline.Word2VecModel
                    print("using {} to read external embedding file {}".format(EmbeddingT, ext_embed))
                    embeddings[key] = EmbeddingT(ext_embed, known_vocab=vocab, unif_weight=unif, keep_unused=False)
                else:
                    print("randomly initializing external feature with dimension {}".format(ext_emb_dsz))
                    embeddings[key] = baseline.RandomInitVecModel(ext_emb_dsz, vocab, unif_weight=unif)
            elif key not in ['word', 'char']:
                raise Exception("Error: must specify a field '{}' in 'extended_embed_sz' dictionary for embedding dim size".format(key))

        out_vocabs = {}
        for key, value in embeddings.items():
            out_vocabs[key] = value.vocab
        return embeddings, out_vocabs

    @staticmethod
    def _read_config(config):
        with open(config) as f:
            return json.load(f)

    @staticmethod
    def _log2json(log):
        s = []
        with open(log) as f:
            for line in f:
                x = line.replace("'", '"')
                s.append(json.loads(x))
        return s

    def create_exporter(self):
        return self.ExporterType(self)


class ClassifierTask(Task):

    def __init__(self, logging_file, **kwargs):
        super(ClassifierTask, self).__init__(logging_file, **kwargs)

        self.task = None

    def _create_task_specific_reader(self):
        return baseline.create_pred_reader(self.config_params['preproc']['mxlen'],
                                           zeropadding=0,
                                           clean_fn=self.config_params['preproc']['clean_fn'],
                                           vec_alloc=self.config_params['preproc']['vec_alloc'],
                                           src_vec_trans=self.config_params['preproc']['src_vec_trans'],
                                           reader_type=self.config_params['loader']['reader_type'])

    def _setup_task(self):
        backend = self.config_params.get('backend', 'tensorflow')
        if backend == 'pytorch':
            print('PyTorch backend')
            from baseline.pytorch import long_0_tensor_alloc
            from baseline.pytorch import tensor_reverse_2nd as rev2nd
            import baseline.pytorch.classify as classify
            self.config_params['preproc']['vec_alloc'] = long_0_tensor_alloc

        else:
            self.config_params['preproc']['vec_alloc'] = np.zeros

            if backend == 'keras':
                print('Keras backend')
                import baseline.keras.classify as classify
            else:
                print('TensorFlow backend')
                import baseline.tf.classify as classify
                from baseline.data import reverse_2nd as rev2nd
                import mead.tf
                self.ExporterType = mead.tf.ClassifyTensorFlowExporter

        self.task = classify

        if self.config_params['preproc'].get('clean', False) is True:
            self.config_params['preproc']['clean_fn'] = baseline.TSVSeqLabelReader.do_clean
            print('Clean')
        elif self.config_params['preproc'].get('lower', False) is True:
            self.config_params['preproc']['clean_fn'] = baseline.lowercase
            print('Lower')
        else:
            self.config_params['preproc']['clean_fn'] = None

        self.config_params['preproc']['src_vec_trans'] = rev2nd if self.config_params['preproc'].get('rev', False) else None
        self.config_params['model']['mxlen'] = self.config_params['preproc']['mxlen']

    def initialize(self, embeddings):
        embeddings_set = mead.utils.index_by_label(embeddings)
        vocab, self.labels = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, {'word': vocab})

    def _create_model(self):
        return self.task.create_model(self.embeddings, self.labels, **self.config_params['model'])

    def _load_dataset(self):
        self.train_data = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'], shuffle=True)
        self.valid_data = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'])
        self.test_data = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params.get('test_batchsz', 1))

Task.TASK_REGISTRY['classify'] = ClassifierTask


class TaggerTask(Task):

    def __init__(self, logging_file, **kwargs):
        super(TaggerTask, self).__init__(logging_file, **kwargs)
        self.task = None

    def _create_task_specific_reader(self):
        preproc = self.config_params['preproc']
        reader = baseline.create_seq_pred_reader(preproc['mxlen'],
                                                 preproc['mxwlen'],
                                                 preproc['word_trans_fn'],
                                                 preproc['vec_alloc'],
                                                 preproc['vec_shape'],
                                                 preproc['trim'],
                                                 **self.config_params['loader'])
        return reader

    def _setup_task(self):
        backend = self.config_params.get('backend', 'tensorflow')
        if backend == 'pytorch':
            print('PyTorch backend')
            from baseline.pytorch import long_0_tensor_alloc as vec_alloc
            from baseline.pytorch import tensor_shape as vec_shape
            import baseline.pytorch.tagger as tagger
            self.config_params['preproc']['vec_alloc'] = vec_alloc
            self.config_params['preproc']['vec_shape'] = vec_shape
            self.config_params['preproc']['trim'] = True
        else:
            self.config_params['preproc']['vec_alloc'] = np.zeros
            self.config_params['preproc']['vec_shape'] = np.shape
            print('TensorFlow backend')
            self.config_params['preproc']['trim'] = False
            import baseline.tf.tagger as tagger
            import mead.tf
            self.ExporterType = mead.tf.TaggerTensorFlowExporter

        self.task = tagger
        if self.config_params['preproc'].get('web-cleanup', False) is True:
            self.config_params['preproc']['word_trans_fn'] = baseline.CONLLSeqReader.web_cleanup
            print('Web-ish data cleanup')
        elif self.config_params['preproc'].get('lower', False) is True:
            self.config_params['preproc']['word_trans_fn'] = baseline.lowercase
            print('Lower')
        else:
            self.config_params['word_trans_fn'] = None

    def initialize(self, embeddings):
        embeddings_set = mead.utils.index_by_label(embeddings)
        vocabs = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocabs)

    def _create_model(self):
        labels = self.reader.label2index
        self.config_params['model']["unif"] = self.config_params["unif"]
        self.config_params['model']['maxs'] = self.reader.max_sentence_length
        self.config_params['model']['maxw'] = self.reader.max_word_length
        return self.task.create_model(labels, self.embeddings, **self.config_params['model'])

    def _load_dataset(self):
        self.train_data, _ = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'], shuffle=True)
        self.valid_data, _ = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'])
        self.test_data, _ = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params.get('test_batchsz', 1), shuffle=False)

Task.TASK_REGISTRY['tagger'] = TaggerTask


class EncoderDecoderTask(Task):

    def __init__(self, logging_file, **kwargs):
        super(EncoderDecoderTask, self).__init__(logging_file, **kwargs)
        self.task = None

    def _create_task_specific_reader(self):
        preproc = self.config_params['preproc']
        reader = baseline.create_parallel_corpus_reader(preproc['mxlen'],
                                                        preproc['vec_alloc'],
                                                        preproc['trim'],
                                                        preproc['word_trans_fn'],
                                                        **self.config_params['loader'])
        return reader

    def _setup_task(self):

        # If its not vanilla seq2seq, dont bother reversing
        do_reverse = self.config_params['model']['model_type'] == 'default'
        backend = self.config_params.get('backend', 'tensorflow')
        if backend == 'pytorch':
            print('PyTorch backend')
            from baseline.pytorch import long_0_tensor_alloc as vec_alloc
            from baseline.pytorch import tensor_shape as vec_shape
            import baseline.pytorch.seq2seq as seq2seq
            self.config_params['preproc']['vec_alloc'] = vec_alloc
            self.config_params['preproc']['vec_shape'] = vec_shape
            src_vec_trans = baseline.tensor_reverse_2nd if do_reverse else None
            self.config_params['preproc']['word_trans_fn'] = src_vec_trans
            self.config_params['preproc']['show_ex'] = baseline.pytorch.show_examples_pytorch
            self.config_params['preproc']['trim'] = True
        else:
            import baseline.tf.seq2seq as seq2seq
            import mead.tf
            self.ExporterType = mead.tf.Seq2SeqTensorFlowExporter
            self.config_params['preproc']['vec_alloc'] = np.zeros
            self.config_params['preproc']['vec_shape'] = np.shape
            self.config_params['preproc']['trim'] = False
            src_vec_trans = baseline.reverse_2nd if do_reverse else None
            self.config_params['preproc']['word_trans_fn'] = src_vec_trans
            self.config_params['preproc']['show_ex'] = baseline.tf.show_examples_tf

        self.task = seq2seq

    def initialize(self, embeddings):
        embeddings_set = mead.utils.index_by_label(embeddings)

        vocab_file = self.dataset.get('vocab_file', None)
        if vocab_file is not None:
            vocab1, vocab2 = self.reader.build_vocabs([vocab_file])
        else:
            vocab1, vocab2 = self.reader.build_vocabs([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])
        self.embeddings1, self.feat2index1 = self._create_embeddings(embeddings_set, {'word': vocab1})
        self.embeddings2, self.feat2index2 = self._create_embeddings(embeddings_set, {'word': vocab2})

    def _load_dataset(self):
        self.train_data = self.reader.load(self.dataset['train_file'], self.feat2index1['word'], self.feat2index2['word'], self.config_params['batchsz'], shuffle=True)
        self.valid_data = self.reader.load(self.dataset['valid_file'], self.feat2index1['word'], self.feat2index2['word'], self.config_params['batchsz'], shuffle=True)
        self.test_data = self.reader.load(self.dataset['test_file'], self.feat2index1['word'], self.feat2index2['word'], self.config_params.get('test_batchsz', 1))

    def _create_model(self):
        return self.task.create_model(self.embeddings1['word'], self.embeddings2['word'], **self.config_params['model'])

    def train(self):

        num_ex = self.config_params['num_valid_to_show']

        if num_ex > 0:
            print('Showing examples')
            preproc = self.config_params['preproc']
            show_ex_fn = preproc['show_ex']
            rlut1 = baseline.revlut(self.feat2index1['word'])
            rlut2 = baseline.revlut(self.feat2index2['word'])
            self.config_params['train']['after_train_fn'] = lambda model: show_ex_fn(model,
                                                                                     self.valid_data, rlut1, rlut2,
                                                                                     self.embeddings2['word'],
                                                                                     preproc['mxlen'], False, 0,
                                                                                     num_ex, reverse=False)
        super(EncoderDecoderTask, self).train()

Task.TASK_REGISTRY['seq2seq'] = EncoderDecoderTask


class LanguageModelingTask(Task):

    def __init__(self, logging_file, **kwargs):
        super(LanguageModelingTask, self).__init__(logging_file, **kwargs)
        self.task = None

    def _create_task_specific_reader(self):
        mxwlen = self.config_params['preproc']['mxwlen']
        nbptt = self.config_params['nbptt']
        reader = baseline.create_lm_reader(mxwlen,
                                           nbptt,
                                           self.config_params['preproc']['word_trans_fn'],
                                           reader_type=self.config_params['loader']['reader_type'])
        return reader

    def _setup_task(self):

        backend = self.config_params.get('backend', 'tensorflow')
        if backend == 'pytorch':
            print('PyTorch backend')
            from baseline.pytorch import long_0_tensor_alloc as vec_alloc
            from baseline.pytorch import tensor_shape as vec_shape
            import baseline.pytorch.lm as lm
            self.config_params['preproc']['vec_alloc'] = vec_alloc
            self.config_params['preproc']['vec_shape'] = vec_shape
            self.config_params['preproc']['trim'] = True
        else:
            self.config_params['preproc']['vec_alloc'] = np.zeros
            self.config_params['preproc']['vec_shape'] = np.shape
            print('TensorFlow backend')
            self.config_params['preproc']['trim'] = False
            import baseline.tf.lm as lm
        self.task = lm

        if self.config_params.get('web-cleanup', False) is True:
            self.config_params['preproc']['word_trans_fn'] = baseline.CONLLSeqReader.web_cleanup
            print('Web-ish data cleanup')
        elif self.config_params.get('lower', False) is True:
            self.config_params['preproc']['word_trans_fn'] = baseline.lowercase
            print('Lower')
        else:
            self.config_params['preproc']['word_trans_fn'] = None

    def initialize(self, embeddings):
        embeddings_set = mead.utils.index_by_label(embeddings)
        vocab, self.num_words = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocab)

    def _load_dataset(self):
        mxwlen = self.config_params['preproc']['mxwlen']
        if mxwlen > 0:
            self.reader.max_word_length = max(mxwlen, self.reader.max_word_length)
        self.train_data = self.reader.load(self.dataset['train_file'], self.feat2index, self.num_words[0], self.config_params['batchsz'])
        self.valid_data = self.reader.load(self.dataset['valid_file'], self.feat2index, self.num_words[1], self.config_params['batchsz'])
        self.test_data = self.reader.load(self.dataset['test_file'], self.feat2index, self.num_words[2], self.config_params['batchsz'])

    def _create_model(self):

        model = self.config_params['model']
        model['unif'] = self.config_params['unif']
        model['batchsz'] = self.config_params['batchsz']
        model['nbptt'] = self.config_params['nbptt']
        model['maxw'] = self.reader.max_word_length
        return self.task.create_model(self.embeddings, **model)

    @staticmethod
    def _num_steps_per_epoch(num_examples, nbptt, batchsz):
        rest = num_examples // batchsz
        return rest // nbptt

    def train(self):
        # TODO: This should probably get generalized and pulled up
        if self.config_params['train'].get('decay_type', None) == 'zaremba':
            batchsz = self.config_params['batchsz']
            nbptt = self.config_params['nbptt']
            steps_per_epoch = LanguageModelingTask._num_steps_per_epoch(self.num_words[0], nbptt, batchsz)
            first_range = int(self.config_params['train']['start_decay_epoch'] * steps_per_epoch)

            self.config_params['train']['bounds'] = [first_range] + list(np.arange(self.config_params['train']['start_decay_epoch'] + 1,
                                                                                   self.config_params['train']['epochs'] + 1,
                                                                                   dtype=np.int32) * steps_per_epoch)

        super(LanguageModelingTask, self).train()

Task.TASK_REGISTRY['lm'] = LanguageModelingTask
