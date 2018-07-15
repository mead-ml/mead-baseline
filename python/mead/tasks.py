import baseline
import json
import numpy as np
import logging
import logging.config
import mead.utils
import os
from mead.downloader import EmbeddingDownloader, DataDownloader
from mead.mime_type import mime_type
from baseline.utils import export, read_config_file, read_json, write_json

__all__ = []
exporter = export(__all__)

@exporter
class Task(object):
    TASK_REGISTRY = {}

    def __init__(self, logger_file, mead_config):
        super(Task, self).__init__()
        self.config_params = None
        self.ExporterType = None
        self.mead_config = mead_config
        if os.path.exists(mead_config):
            mead_settings = read_json(mead_config)
        else:
            mead_settings = {}
        if 'datacache' not in mead_settings:
            self.data_download_cache = os.path.expanduser("~/.bl-data")
            mead_settings['datacache'] = self.data_download_cache
            write_json(mead_settings, mead_config)
        else:
            self.data_download_cache = os.path.expanduser(mead_settings['datacache'])
        print("using {} as data/embeddings cache".format(self.data_download_cache))
        self._configure_logger(logger_file)

    def _configure_logger(self, logger_file):
        """Use the logger file (logging.json) to configure the log, but overwrite the filename to include the PID

        :param logger_file: The logging configuration JSON file
        :return: A dictionary config derived from the logger_file, with the reporting handler suffixed with PID
        """
        with open(logger_file) as f:
            config = json.load(f)
            config['handlers']['reporting_file_handler']['filename'] = 'reporting-{}.log'.format(os.getpid())
            logging.config.dictConfig(config)

    @staticmethod
    def get_task_specific(task, logging_config, mead_config):
        """Get the task from the task registry associated with the name

        :param task: The task name
        :param logging_config: The configuration to read from
        :return:
        """
        config = Task.TASK_REGISTRY[task](logging_config, mead_config)
        return config

    def read_config(self, config_params, datasets_index):
        """
        Read the config file and the datasets index

        Between the config file and the dataset index, we have enough information
        to configure the backend and the models.  We can also initialize the data readers

        :param config_file: The config file
        :param datasets_index: The index of datasets
        :return:
        """
        datasets_set = mead.utils.index_by_label(datasets_index)
        self.config_params = config_params
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
    def _create_embeddings_from_file(embed_file, embed_dsz, embed_sha1, data_download_cache, vocab, unif, keep_unused):
        embed_file = EmbeddingDownloader(embed_file, embed_dsz, embed_sha1, data_download_cache).download()
        EmbeddingT = baseline.GloVeModel if mime_type(embed_file) == 'text/plain' else baseline.Word2VecModel
        return EmbeddingT(embed_file, vocab, unif_weight=unif, keep_unused=keep_unused)

    def _create_embeddings(self, embeddings_set, vocabs):

        unif = self.config_params['unif']
        keep_unused = self.config_params.get('keep_unused', False)

        if 'word' in vocabs:
            embeddings_section = self.config_params['word_embeddings']
            embed_label = embeddings_section.get('label', None)

            embeddings = dict()
            if embed_label is not None:
                embed_file = embeddings_set[embed_label]['file']
                embed_dsz = embeddings_set[embed_label]['dsz']
                embed_sha1 = embeddings_set[embed_label].get('sha1',None)
                embeddings['word'] = Task._create_embeddings_from_file(embed_file, embed_dsz, embed_sha1,
                                                                       self.data_download_cache, vocabs['word'],
                                                                       unif=unif, keep_unused=keep_unused)
            else:
                dsz = embeddings_section['dsz']
                embeddings['word'] = baseline.RandomInitVecModel(dsz, vocabs['word'], unif_weight=unif)

        if 'char' in vocabs:
            if self.config_params.get('charsz', -1) > 0:
                embeddings['char'] = baseline.RandomInitVecModel(self.config_params['charsz'], vocabs['char'], unif_weight=unif)

        extended_embed_info = self.config_params.get('extended_embed_info', {})
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
    def _log2json(log):
        s = []
        with open(log) as f:
            for line in f:
                x = line.replace("'", '"')
                s.append(json.loads(x))
        return s

    def create_exporter(self):
        return self.ExporterType(self)


@exporter
class ClassifierTask(Task):

    def __init__(self, logging_file, mead_config, **kwargs):
        super(ClassifierTask, self).__init__(logging_file, mead_config, **kwargs)
        self.task = None

    def _create_task_specific_reader(self):
        return baseline.create_pred_reader(self.config_params['preproc']['mxlen'],
                                           zeropadding=0,
                                           clean_fn=self.config_params['preproc']['clean_fn'],
                                           vec_alloc=self.config_params['preproc']['vec_alloc'],
                                           src_vec_trans=self.config_params['preproc']['src_vec_trans'],
                                           mxwlen=self.config_params['preproc'].get('mxwlen', -1),
                                           trim=self.config_params['preproc'].get('trim', False),
                                           **self.config_params['loader'])

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
            if backend == 'dynet':
                print('Dynet backend')
                import _dynet
                dy_params = _dynet.DynetParams()
                dy_params.from_args()
                dy_params.set_requested_gpus(1)
                if 'autobatchsz' in self.config_params['train']:
                    self.config_params['model']['batched'] = False
                    dy_params.set_autobatch(True)
                dy_params.init()
                import baseline.dy.classify as classify
                from baseline.data import reverse_2nd as rev2nd
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

    def initialize(self, embeddings):
        embeddings_set = mead.utils.index_by_label(embeddings)
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']))
        vocab, self.labels = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocab)


    def _create_model(self):
        model = self.config_params['model']
        model['mxlen'] = self.reader.max_sentence_length
        model['mxwlen'] = self.reader.max_word_length
        return self.task.create_model(self.embeddings, self.labels, **model)

    def _load_dataset(self):
        self.train_data = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'], shuffle=True)
        self.valid_data = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'])
        self.test_data = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params.get('test_batchsz', 1))

Task.TASK_REGISTRY['classify'] = ClassifierTask


@exporter
class TaggerTask(Task):

    def __init__(self, logging_file, mead_config, **kwargs):
        super(TaggerTask, self).__init__(logging_file, mead_config, **kwargs)
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
        elif backend == 'dynet':
            print('Dynet backend')
            import _dynet
            dy_params = _dynet.DynetParams()
            dy_params.from_args()
            dy_params.set_requested_gpus(1)
            dy_params.set_requested_gpus(1)
            if 'autobatchsz' in self.config_params['train']:
                self.config_params['model']['batched'] = False
                dy_params.set_autobatch(True)
            else:
                raise Exception('Tagger currently only supports autobatching.'
                                'Change "batchsz" to 1 and under "train", set "autobatchsz" to your desired batchsz')
                #self.config_params['model']['batched'] = True
                #dy_params.set_autobatch(False)
            dy_params.init()

            dy_params.init()
            import baseline.dy.tagger as tagger
            self.config_params['preproc']['vec_alloc'] = np.zeros
            self.config_params['preproc']['vec_shape'] = np.shape
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
            self.config_params['preproc']['word_trans_fn'] = None

    def initialize(self, embeddings):
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']))
        embeddings_set = mead.utils.index_by_label(embeddings)
        vocabs = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocabs)

    def _create_model(self):
        labels = self.reader.label2index
        self.config_params['model']['span_type'] = self.config_params['train'].get('span_type')
        self.config_params['model']["unif"] = self.config_params["unif"]
        self.config_params['model']['maxs'] = self.reader.max_sentence_length
        self.config_params['model']['maxw'] = self.reader.max_word_length
        return self.task.create_model(labels, self.embeddings, **self.config_params['model'])

    def _load_dataset(self):
        self.train_data, _ = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'], shuffle=True)
        self.valid_data, _ = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'])
        self.test_data, self.txts = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params.get('test_batchsz', 1), shuffle=False, do_sort=False)

    def train(self):
        self._load_dataset()
        model = self._create_model()
        conll_output = self.config_params.get("conll_output", None)
        self.task.fit(model, self.train_data, self.valid_data, self.test_data, conll_output=conll_output, txts=self.txts, **self.config_params['train'])
        return model

Task.TASK_REGISTRY['tagger'] = TaggerTask


@exporter
class EncoderDecoderTask(Task):

    def __init__(self, logging_file, mead_config, **kwargs):
        super(EncoderDecoderTask, self).__init__(logging_file, mead_config, **kwargs)
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
            from baseline.pytorch import tensor_reverse_2nd as rev2nd
            import baseline.pytorch.seq2seq as seq2seq
            self.config_params['preproc']['vec_alloc'] = vec_alloc
            self.config_params['preproc']['vec_shape'] = vec_shape
            src_vec_trans = rev2nd if do_reverse else None
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
        self.dataset = DataDownloader(self.dataset, self.data_download_cache, True).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}\n[vocab file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file'], self.dataset.get('vocab_file',"None")))
        vocab_file = self.dataset.get('vocab_file',None)
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


@exporter
class LanguageModelingTask(Task):

    def __init__(self, logging_file, mead_config, **kwargs):
        super(LanguageModelingTask, self).__init__(logging_file, mead_config, **kwargs)
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
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']))
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
