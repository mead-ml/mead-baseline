import baseline
import json
import logging
import logging.config
import mead.utils
import os
from mead.downloader import EmbeddingDownloader, DataDownloader
from baseline.utils import (export, read_json, zip_files, save_vectorizers, save_vocabs)
from baseline.vectorizers import create_vectorizer
__all__ = []
exporter = export(__all__)


@exporter
class Task(object):
    TASK_REGISTRY = {}

    def __init__(self, logger_file, mead_settings_config=None):
        super(Task, self).__init__()
        self.config_params = None
        self.ExporterType = None
        if mead_settings_config is None:
            self.mead_settings_config = {}
        elif isinstance(mead_settings_config, dict):
            self.mead_settings_config = mead_settings_config
        elif os.path.exists(mead_settings_config):
            self.mead_settings_config = read_json(mead_settings_config)
        else:
            raise Exception("Expected either a mead settings file or a JSON object")
        if 'datacache' not in self.mead_settings_config:
            self.data_download_cache = os.path.expanduser("~/.bl-data")
            self.mead_settings_config['datacache'] = self.data_download_cache
        else:
            self.data_download_cache = os.path.expanduser(self.mead_settings_config['datacache'])
        print("using {} as data/embeddings cache".format(self.data_download_cache))
        self._configure_logger(logger_file)

    def _create_vectorizers(self):
        self.vectorizers = {}
        for feature in self.config_params['features']:
            key = feature['name']
            vectorizer_section = feature.get('vectorizer', {'type': 'token1d'})
            vectorizer_section['mxlen'] = vectorizer_section.get('mxlen', self.config_params['preproc'].get('mxlen', -1))
            vectorizer_section['mxwlen'] = vectorizer_section.get('mxlen', self.config_params['preproc'].get('mxwlen', -1))
            if 'transform' in vectorizer_section:
                vectorizer_section['transform_fn'] = eval(vectorizer_section['transform'])
            vectorizer = create_vectorizer(**vectorizer_section)
            self.vectorizers[key] = vectorizer

    def _configure_logger(self, logger_config):
        """Use the logger file (logging.json) to configure the log, but overwrite the filename to include the PID

        :param logger_file: The logging configuration JSON file
        :return: A dictionary config derived from the logger_file, with the reporting handler suffixed with PID
        """
        if isinstance(logger_config, dict):
            config = logger_config
        elif os.path.exists(logger_config):
            config = read_json(logger_config)
        else:
            raise Exception("Expected logger config file or a JSON object")

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
        basedir = self.config_params.get('basedir')
        if basedir is not None and not os.path.exists(basedir):
            print('Creating: {}'.format(basedir))
            os.mkdir(basedir)
        self.config_params['train']['basedir'] = basedir
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
        save_vectorizers(self.get_basedir(), self.vectorizers)
        model = self._create_model()
        self.task.fit(model, self.train_data, self.valid_data, self.test_data, **self.config_params['train'])
        zip_files(self.get_basedir())
        return model

    def _configure_reporting(self):
        reporting = {
            "logging": True,
            "visdom": self.config_params.get('visdom', False),
            "tensorboard": self.config_params.get('tensorboard', False),
            "visdom_name": self.config_params.get('visdom_name', 'main'),
        }
        reporting = baseline.setup_reporting(**reporting)
        self.config_params['train']['reporting'] = reporting
        logging.basicConfig(level=logging.DEBUG)

    def _create_embeddings(self, embeddings_set, vocabs, features):
        backend = self.config_params.get('backend', 'tensorflow')

        if backend == 'pytorch':
            import baseline.pytorch.embeddings as embeddings
        elif backend == 'keras':
            print('Keras backend')
            import baseline.keras.embeddings as embeddings
        elif backend == 'dynet':
            print('Dynet backend')
            import baseline.dy.embeddings as embeddings
        else:
            print('TensorFlow backend')
            import baseline.tf.embeddings as embeddings

        unif = self.config_params['unif']
        keep_unused = self.config_params.get('keep_unused', False)

        embeddings_map = dict()
        out_vocabs = {}
        for feature in features:
            embeddings_section = feature['embeddings']
            name = feature['name']
            embed_label = embeddings_section.get('label', None)
            embed_type = embeddings_section.get('type', 'default')
            embeddings_section['unif'] = embeddings_section.get('unif', unif)
            embeddings_section['keep_unused'] = embeddings_section.get('keep_unused', keep_unused)
            if embed_label is not None:
                # Allow local overrides to uniform initializer

                embed_file = embeddings_set[embed_label]['file']
                embed_dsz = embeddings_set[embed_label]['dsz']
                embed_sha1 = embeddings_set[embed_label].get('sha1', None)
                embed_file = EmbeddingDownloader(embed_file, embed_dsz, embed_sha1, self.data_download_cache).download()
                embedding_bundle = embeddings.load_embeddings(embed_file,
                                                              name,
                                                              known_vocab=vocabs[name],
                                                              embed_type=embed_type,
                                                              **embeddings_section)

                embeddings_map[name] = embedding_bundle['embeddings']
                out_vocabs[name] = embedding_bundle['vocab']
            else:
                dsz = embeddings_section.pop('dsz')
                embedding_bundle = embeddings.create_embeddings(dsz, name, vocabs[name], embed_type=embed_type, **embeddings_section)
                embeddings_map[name] = embedding_bundle['embeddings']
                out_vocabs[name] = embedding_bundle['vocab']

        return embeddings_map, out_vocabs

    @staticmethod
    def _log2json(log):
        s = []
        with open(log) as f:
            for line in f:
                x = line.replace("'", '"')
                s.append(json.loads(x))
        return s

    def get_basedir(self):
        return self.config_params.get('basedir', './')


@exporter
class ClassifierTask(Task):

    def __init__(self, logging_file, mead_settings_config, **kwargs):
        super(ClassifierTask, self).__init__(logging_file, mead_settings_config, **kwargs)
        self.task = None

    def _create_task_specific_reader(self):
        self._create_vectorizers()
        return baseline.create_pred_reader(self.vectorizers, clean_fn=self.config_params['preproc']['clean_fn'],
                                           trim=self.config_params['preproc'].get('trim', False),
                                           **self.config_params['loader'])

    def _setup_task(self):
        backend = self.config_params.get('backend', 'tensorflow')
        if backend == 'pytorch':
            print('PyTorch backend')
            import baseline.pytorch.classify as classify
        elif backend == 'keras':
            print('Keras backend')
            import baseline.keras.classify as classify
        elif backend == 'dynet':
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
        else:
            print('TensorFlow backend')
            import baseline.tf.classify as classify
            from mead.tf.exporters import ClassifyTensorFlowExporter
            self.ExporterType = ClassifyTensorFlowExporter

        self.task = classify

        if self.config_params['preproc'].get('clean', False) is True:
            self.config_params['preproc']['clean_fn'] = baseline.TSVSeqLabelReader.do_clean
            print('Clean')
        else:
            self.config_params['preproc']['clean_fn'] = None

    def initialize(self, embeddings):
        embeddings_set = mead.utils.index_by_label(embeddings)
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']))
        vocab, self.labels = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocab, self.config_params['features'])
        save_vocabs(self.get_basedir(), self.feat2index)

    def _create_model(self):
        model = self.config_params['model']
        lengths_key = model.get('lengths_key', self.config_params.get('sort_key'))
        if lengths_key is not None:
            if not lengths_key.endswith('_lengths'):
                lengths_key = '{}_lengths'.format(lengths_key)
            model['lengths_key'] = lengths_key
        return self.task.create_model(self.embeddings, self.labels, **model)

    def _load_dataset(self):
        self.train_data = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'], shuffle=True, sort_key=self.config_params['loader'].get('sort_key'))
        self.valid_data = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'])
        self.test_data = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params.get('test_batchsz', 1))

Task.TASK_REGISTRY['classify'] = ClassifierTask


@exporter
class TaggerTask(Task):

    def __init__(self, logging_file, mead_config, **kwargs):
        super(TaggerTask, self).__init__(logging_file, mead_config, **kwargs)
        self.task = None

    def _create_task_specific_reader(self):
        self._create_vectorizers()
        return baseline.create_seq_pred_reader(self.vectorizers, trim=self.config_params['preproc'].get('trim', False),
                                               **self.config_params['loader'])

    def _setup_task(self):
        backend = self.config_params.get('backend', 'tensorflow')
        if backend == 'pytorch':
            print('PyTorch backend')
            import baseline.pytorch.tagger as tagger
            self.config_params['preproc']['trim'] = True
        elif backend == 'dynet':
            print('Dynet backend')
            import _dynet
            dy_params = _dynet.DynetParams()
            dy_params.from_args()
            dy_params.set_requested_gpus(1)
            if 'autobatchsz' in self.config_params['train']:
                self.config_params['model']['batched'] = False
                dy_params.set_autobatch(True)
            else:
                raise Exception('Tagger currently only supports autobatching.'
                                'Change "batchsz" to 1 and under "train", set "autobatchsz" to your desired batchsz')
            dy_params.init()
            import baseline.dy.tagger as tagger
            self.config_params['preproc']['trim'] = True
        else:
            print('TensorFlow backend')
            self.config_params['preproc']['trim'] = False
            import baseline.tf.tagger as tagger
            from mead.tf.exporters import TaggerTensorFlowExporter
            self.ExporterType = TaggerTensorFlowExporter
        self.task = tagger

    def initialize(self, embeddings):
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']))
        embeddings_set = mead.utils.index_by_label(embeddings)
        vocabs = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocabs, self.config_params['features'])
        save_vocabs(self.get_basedir(), self.feat2index)

    def _create_model(self):
        """
        if model.lengths_key is None:
            if 'word' in model.embeddings:
                model.lengths_key = 'word'
            elif 'x' in model.embeddings:
                model.lengths_key = 'x'

        if model.lengths_key is None:
            raise Exception("Require a `lengths_key`")
            # This allows user to short-hand the field to use
        if not model.lengths_key.endswith('_lengths'):
            model.lengths_key += '_lengths'
        :return:
        """
        labels = self.reader.label2index
        self.config_params['model']['span_type'] = self.config_params['train'].get('span_type')
        self.config_params['model']["unif"] = self.config_params["unif"]
        model = self.config_params['model']
        lengths_key = model.get('lengths_key', self.config_params.get('sort_key', 'word'))
        if lengths_key is not None:
            if not lengths_key.endswith('_lengths'):
                lengths_key = '{}_lengths'.format(lengths_key)
            model['lengths_key'] = lengths_key


        return self.task.create_model(labels, self.embeddings, **self.config_params['model'])

    def _load_dataset(self):
        self.train_data, _ = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'], shuffle=True, sort_key=self.config_params['loader'].get('sort_key', 'y'))
        self.valid_data, _ = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'], sort_key=None)
        self.test_data, self.txts = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params.get('test_batchsz', 1), shuffle=False, sort_key=None)

    def train(self):
        self._load_dataset()
        save_vectorizers(self.get_basedir(), self.vectorizers)
        model = self._create_model()
        conll_output = self.config_params.get("conll_output", None)
        self.task.fit(model, self.train_data, self.valid_data, self.test_data, conll_output=conll_output, txts=self.txts, **self.config_params['train'])
        zip_files(self.get_basedir())
        return model

Task.TASK_REGISTRY['tagger'] = TaggerTask


@exporter
class EncoderDecoderTask(Task):

    def __init__(self, logging_file, mead_config, **kwargs):
        super(EncoderDecoderTask, self).__init__(logging_file, mead_config, **kwargs)
        self.task = None

    def _create_task_specific_reader(self):
        self._create_vectorizers()
        preproc = self.config_params['preproc']
        reader = baseline.create_parallel_corpus_reader(self.vectorizers,
                                                        preproc['trim'],
                                                        **self.config_params['loader'])
        return reader

    def _setup_task(self):

        backend = self.config_params.get('backend', 'tensorflow')
        if backend == 'pytorch':
            print('PyTorch backend')
            import baseline.pytorch.seq2seq as seq2seq
            self.config_params['preproc']['show_ex'] = baseline.pytorch.show_examples_pytorch
            self.config_params['preproc']['trim'] = True
        else:
            # TODO: why not support DyNet trimming?
            self.config_params['preproc']['trim'] = False
            if backend == 'dynet':
                print('Dynet backend')
                import _dynet
                dy_params = _dynet.DynetParams()
                dy_params.from_args()
                dy_params.set_requested_gpus(1)
                dy_params.init()
                import baseline.dy.seq2seq as seq2seq
                self.config_params['preproc']['show_ex'] = baseline.dy.show_examples_dynet
            else:
                import baseline.tf.seq2seq as seq2seq
                sort_key = self.config_params['loader'].get('src_sort_key', 'src')
                self.config_params['preproc']['show_ex'] = baseline.tf.create_show_examples_tf(sort_key)
                from mead.tf.exporters import Seq2SeqTensorFlowExporter
                self.ExporterType = Seq2SeqTensorFlowExporter

        self.task = seq2seq

    def initialize(self, embeddings):
        embeddings_set = mead.utils.index_by_label(embeddings)
        self.dataset = DataDownloader(self.dataset, self.data_download_cache, True).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}\n[vocab file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file'], self.dataset.get('vocab_file',"None")))
        vocab_file = self.dataset.get('vocab_file')
        if vocab_file is not None:
            vocab1, vocab2 = self.reader.build_vocabs([vocab_file])
        else:
            vocab1, vocab2 = self.reader.build_vocabs([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])

        # To keep the config file simple, share a list between source and destination (tgt)
        features_src = []
        features_tgt = None
        for feature in self.config_params['features']:
            if feature['name'] == 'tgt':
                features_tgt = feature
            else:
                features_src += [feature]

        self.src_embeddings, self.feat2src = self._create_embeddings(embeddings_set, vocab1, features_src)
        # For now, dont allow multiple vocabs of output
        save_vocabs(self.get_basedir(), self.feat2src)
        self.tgt_embeddings, self.feat2tgt = self._create_embeddings(embeddings_set, {'tgt': vocab2}, [features_tgt])
        save_vocabs(self.get_basedir(), self.feat2tgt)
        self.tgt_embeddings = self.tgt_embeddings['tgt']
        self.feat2tgt = self.feat2tgt['tgt']

    def _load_dataset(self):
        sort_key = self.config_params['loader'].get('src_sort_key', 'src')
        self.train_data = self.reader.load(self.dataset['train_file'],
                                           self.feat2src, self.feat2tgt,
                                           self.config_params['batchsz'],
                                           shuffle=True, sort_key=sort_key)
        self.valid_data = self.reader.load(self.dataset['valid_file'],
                                           self.feat2src,
                                           self.feat2tgt,
                                           self.config_params['batchsz'],
                                           shuffle=True)
        self.test_data = self.reader.load(self.dataset['test_file'],
                                          self.feat2src,
                                          self.feat2tgt,
                                          self.config_params.get('test_batchsz', 1))

    def _create_model(self):
        self.config_params['model']['GO'] = self.feat2tgt['<GO>']
        self.config_params['model']['EOS'] = self.feat2tgt['<EOS>']
        self.config_params['model']["unif"] = self.config_params["unif"]
        model = self.config_params['model']
        lengths_key = model.get('src_lengths_key', self.config_params.get('sort_key', 'src'))
        if lengths_key is not None:
            if not lengths_key.endswith('_lengths'):
                lengths_key = '{}_lengths'.format(lengths_key)
            model['src_lengths_key'] = lengths_key

        return self.task.create_model(self.src_embeddings, self.tgt_embeddings, **self.config_params['model'])

    def train(self):

        num_ex = self.config_params['num_valid_to_show']

        if num_ex > 0:
            print('Showing examples')
            preproc = self.config_params['preproc']
            show_ex_fn = preproc['show_ex']
            vocab_name = self.config_params['loader'].get('src_sort_key', 'src')
            rlut1 = baseline.revlut(self.feat2src[vocab_name])
            rlut2 = baseline.revlut(self.feat2tgt)
            self.config_params['train']['after_train_fn'] = lambda model: show_ex_fn(model,
                                                                                     self.valid_data, rlut1, rlut2,
                                                                                     self.feat2tgt,
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
        self._create_vectorizers()
        nbptt = self.config_params['nbptt']
        reader = baseline.create_lm_reader(self.vectorizers,
                                           nbptt,
                                           reader_type=self.config_params['loader']['reader_type'])
        return reader

    def _setup_task(self):

        backend = self.config_params.get('backend', 'tensorflow')
        if backend == 'pytorch':
            print('PyTorch backend')
            import baseline.pytorch.lm as lm
            self.config_params['preproc']['trim'] = True

        elif backend == 'dynet':
            print('Dynet backend')
            import _dynet
            dy_params = _dynet.DynetParams()
            dy_params.from_args()
            dy_params.set_requested_gpus(1)
            dy_params.init()
            self.config_params['preproc']['trim'] = False
            import baseline.dy.lm as lm
        else:
            print('TensorFlow backend')
            self.config_params['preproc']['trim'] = False
            import baseline.tf.lm as lm
        self.task = lm

    def initialize(self, embeddings):
        embeddings_set = mead.utils.index_by_label(embeddings)
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']))
        vocabs = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']])
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocabs, self.config_params['features'])
        save_vocabs(self.get_basedir(), self.feat2index)

    def _load_dataset(self):
        tgt_key = self.config_params['loader'].get('tgt_key', 'x')

        self.train_data = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'], tgt_key=tgt_key)
        self.valid_data = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'], tgt_key=tgt_key)
        self.test_data = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params['batchsz'], tgt_key=tgt_key)

    def _create_model(self):

        model = self.config_params['model']
        model['unif'] = self.config_params['unif']
        model['batchsz'] = self.config_params['batchsz']
        model['tgt_key'] = self.config_params['loader'].get('tgt_key', 'x')
        return self.task.create_model(self.embeddings, **model)

    @staticmethod
    def _num_steps_per_epoch(num_examples, nbptt, batchsz):
        rest = num_examples // batchsz
        return rest // nbptt

    def train(self):
        # TODO: This should probably get generalized and pulled up
        #if self.config_params['train'].get('decay_type', None) == 'zaremba':
        #    batchsz = self.config_params['batchsz']
        #    nbptt = self.config_params['nbptt']
        #    steps_per_epoch = LanguageModelingTask._num_steps_per_epoch(self.num_elems[0], nbptt, batchsz)
        #    first_range = int(self.config_params['train']['start_decay_epoch'] * steps_per_epoch)

        #    self.config_params['train']['bounds'] = [first_range] + list(np.arange(self.config_params['train']['start_decay_epoch'] + 1,
        #                                                                           self.config_params['train']['epochs'] + 1,
        #                                                                           dtype=np.int32) * steps_per_epoch)

        super(LanguageModelingTask, self).train()

Task.TASK_REGISTRY['lm'] = LanguageModelingTask
