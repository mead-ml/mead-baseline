import os
import json
import logging
import logging.config
import numpy as np
import baseline
from baseline.utils import export, import_user_module, Offsets
from mead.downloader import EmbeddingDownloader, DataDownloader
from mead.utils import (
    index_by_label,
    normalize_backend,
    get_mead_settings,
    read_config_file_or_json,
)


__all__ = []
exporter = export(__all__)


class Backend(object):
    """Simple object to represent a deep-learning framework backend
    """
    def __init__(self, name=None, params=None, exporter=None):
        """Initialize the backend, optional with constructor args

        :param name: (``str``) Name of the framework: currently one of (`tensorflow`, `pytorch`, `dynet`, `keras`)
        :param params: (``dict``) A dictionary of framework-specific user-data to pass through keyword args to each sub-module
        :param exporter: A framework-specific exporter to facilitate exporting to runtime deployment
        """
        self.name = normalize_backend(name)
        self.params = params
        self.exporter = exporter

    def load(self, task_name):
        base_pkg_name = 'baseline.{}'.format(self.name)
        mod = import_user_module(base_pkg_name)
        import_user_module('{}.embeddings'.format(base_pkg_name))
        import_user_module('{}.{}'.format(base_pkg_name, task_name))
        self.transition_mask = mod.transition_mask


TASK_REGISTRY = {}
@exporter
def register_task(cls):
    TASK_REGISTRY[cls.task_name()] = cls
    return cls


@exporter
class Task(object):
    """Basic building block for a task of NLP problems, e.g. `tagger`, `classify`, etc.
    """

    def _create_backend(self):
        """This method creates and returns a `Backend` object

        :return:
        """
        pass

    def __init__(self, logger_config, mead_settings_config=None):
        super(Task, self).__init__()
        self.config_params = None
        self.mead_settings_config = get_mead_settings(mead_settings_config)
        if 'datacache' not in self.mead_settings_config:
            self.data_download_cache = os.path.expanduser("~/.bl-data")
            self.mead_settings_config['datacache'] = self.data_download_cache
        else:
            self.data_download_cache = os.path.expanduser(self.mead_settings_config['datacache'])
        print("using {} as data/embeddings cache".format(self.data_download_cache))
        self._configure_logger(logger_config)

    @classmethod
    def task_name(cls):
        """This classmethod returns the official name of this task, e.g., `classify` for classification

        :return: (``str``) - String name of this task
        """
        pass

    def _create_vectorizers(self):
        """Read the `features` section of the mead config.  This sections contains both embedding info and vectorizers
        Then use the vectorizer sub-section to instantiate the vectorizers and return them in a ``dict`` with name
        keyed off of the `features->name` and value of `vectorizer`

        :return: (``dict``) - A dictionary of the vectorizers keyed by feature name
        """
        self.vectorizers = {}

        features = self.config_params['features']
        self.primary_key = features[0]['name']
        for feature in self.config_params['features']:
            key = feature['name']
            if feature.get('primary', False) is True:
                self.primary_key = key
            vectorizer_section = feature.get('vectorizer', {'type': 'token1d'})
            vectorizer_section['mxlen'] = vectorizer_section.get('mxlen', self.config_params['preproc'].get('mxlen', -1))
            vectorizer_section['mxwlen'] = vectorizer_section.get('mxlen', self.config_params['preproc'].get('mxwlen', -1))
            if 'transform' in vectorizer_section:
                vectorizer_section['transform_fn'] = eval(vectorizer_section['transform'])
            vectorizer = baseline.create_vectorizer(**vectorizer_section)
            self.vectorizers[key] = vectorizer

    def _configure_logger(self, logger_config):
        """Use the logger file (logging.json) to configure the log, but overwrite the filename to include the PID

        There are reporting and timing loggers that are configured, the latter being used for speed testing.

        :param logger_config: The logging configuration JSON or file containing JSON
        :return: A dictionary config derived from the logger_file, with the reporting handler suffixed with PID
        """

        config = read_config_file_or_json(logger_config, 'logger')
        config['handlers']['reporting_file_handler']['filename'] = 'reporting-{}.log'.format(os.getpid())
        config['handlers']['timing_file_handler']['filename'] = 'timing-{}.log'.format(os.getpid())
        logging.config.dictConfig(config)

    @staticmethod
    def get_task_specific(task, logging_config, mead_config):
        """Get the task from the task registry associated with the name

        :param task: The task name
        :param logging_config: The configuration to read from
        :return:
        """
        config = TASK_REGISTRY[task](logging_config, mead_config)
        return config

    def read_config(self, config_params, datasets_index, **kwargs):
        """
        Read the config file and the datasets index

        Between the config file and the dataset index, we have enough information
        to configure the backend and the models.  We can also initialize the data readers

        :param config_file: The config file
        :param datasets_index: The index of datasets
        :return:
        """
        datasets_index = read_config_file_or_json(datasets_index, 'datasets')
        datasets_set = index_by_label(datasets_index)
        self.config_params = config_params
        basedir = self.get_basedir()
        if basedir is not None and not os.path.exists(basedir):
            print('Creating: {}'.format(basedir))
            os.mkdir(basedir)
        self.config_params['train']['basedir'] = basedir
        self.config_file = kwargs.get('config_file')
        self._setup_task()
        self._load_user_modules()
        self._configure_reporting(config_params.get('reporting', {}), **kwargs)
        self.dataset = datasets_set[self.config_params['dataset']]
        self.reader = self._create_task_specific_reader()

    def _load_user_modules(self):
        if 'modules' in self.config_params:
            for addon in self.config_params['modules']:
                import_user_module(addon)

    def initialize(self, embeddings_index):
        """
        Load the vocabulary using the readers and then load any embeddings required

        :param embeddings_index: The index of embeddings
        :return:
        """
        pass

    def _create_task_specific_reader(self):
        self._create_vectorizers()
        reader_params = self.config_params['loader']
        reader_params['clean_fn'] = reader_params.get('clean_fn', self.config_params['preproc'].get('clean_fn'))
        reader_params['mxlen'] = self.vectorizers[self.primary_key].mxlen
        return baseline.reader.create_reader(self.task_name(), self.vectorizers, self.config_params['preproc'].get('trim', False), **reader_params)

    @staticmethod
    def _get_min_f(config):
        backoff = config['loader'].get('min_f', config.get('preproc', {}).get('min_f', -1))
        return {f['name']: f.get('min_f', backoff) for f in config['features']}

    def _setup_task(self):
        """
        This method provides the task-specific setup
        :return:
        """
        self.backend = self._create_backend()

    def _load_dataset(self):
        """This hook is responsible for creating and initializing the ``DataFeed`` objects to be used for train, dev
        and test phases.  This method should yield a `self.train_data`, `self.valid_data` and `self.test_data` on this
        class

        :return: Nothing
        """
        pass

    def _create_model(self):
        """This hook create the model used for training, using the `model` section of the mead config.  The model is
        returned, not stored as a field of the class

        :return: A representation
        """
        pass

    def train(self):
        """This method delegates to several sub-hooks in order to complete training.

        1. call `_load_dataset()` which initializes the `DataFeed` fields of this class
        2. call `baseline.save_vectorizers()` which write out the bound `vectorizers` fields to a file in the `basedir`
        3. call `baseline.train.fit()` which executes the training procedure and  yields a saved model
        4. call `baseline.zip_files()` which zips all files in the `basedir` with the same `PID` as this process
        5. call `_close_reporting_hooks()` which lets the reporting hooks know that the job is finished
        :return: Nothing
        """
        self._load_dataset()
        baseline.save_vectorizers(self.get_basedir(), self.vectorizers)
        model = self._create_model()
        baseline.train.fit(model, self.train_data, self.valid_data, self.test_data, **self.config_params['train'])
        baseline.zip_files(self.get_basedir())
        self._close_reporting_hooks()

    def _configure_reporting(self, reporting, **kwargs):
        """Configure all `reporting_hooks` specified in the mead settings or overridden at the command line

        :param reporting:
        :param kwargs:
        :return:
        """
        default_reporting = self.mead_settings_config.get('reporting_hooks', {})
        # Add default reporting information to the reporting settings.
        for report_type in default_reporting:
            if report_type in reporting:
                for report_arg, report_val in default_reporting[report_type].items():
                    if report_arg not in reporting[report_type]:
                        reporting[report_type][report_arg] = report_val
        reporting_hooks = list(reporting.keys())
        for settings in reporting.values():
            try:
                import_user_module(settings.get('module', ''))
            except (ImportError, ValueError):
                pass

        self.reporting = baseline.create_reporting(reporting_hooks,
                                                   reporting,
                                                   {'config_file': self.config_file, 'task': self.__class__.task_name(), 'base_dir': self.get_basedir()})

        self.config_params['train']['reporting'] = [x.step for x in self.reporting]
        logging.basicConfig(level=logging.DEBUG)

    def _close_reporting_hooks(self):
        """Tell all reporting objects they are done

        :return: Nothing
        """
        for x in self.reporting:
            x.done()

    def _create_embeddings(self, embeddings_set, vocabs, features):
        """Creates a set of arbitrary sub-graph, DL-framework-specific embeddings by delegating to wired sub-module.

        As part of this process, we take in an index of embeddings by name, a ``dict`` of ``Counter`` objects (keyed by
        feature name), containing the number of times each token has been seen, and a `features` list which is a
        sub-section of the mead config containing the `embeddings` section for each feature.
        This method's job is to either create a sub-graph from a pretrained model, or to create a new random
        initialized sub-graph, taking into account the input vocabulary counters.  The embeddings model has control
        to determine the actual word indices and sub-graph for the embeddings, both of which are returned from this
        method.  If some sort of feature selection is
        performed, such as low count removal that would be required via the delegated methods

        :param embeddings_set: The embeddings index passed to mead driver
        :param vocabs: A set of known ``Counter``s for each vocabulary consisting of a token key and count for each
        :param features: The `features` sub-section of the mead config
        :return: Returns a ``tuple`` comprised of a ``dict`` of (`feature name`, `Embedding`) and an updated vocab
        """
        unif = self.config_params.get('unif', 0.1)
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
            if self.backend.params is not None:
                for k, v in self.backend.params.items():
                    embeddings_section[k] = v
            if embed_label is not None:
                # Allow local overrides to uniform initializer

                embed_file = embeddings_set[embed_label]['file']
                embed_dsz = embeddings_set[embed_label]['dsz']
                embed_sha1 = embeddings_set[embed_label].get('sha1', None)
                embed_file = EmbeddingDownloader(embed_file, embed_dsz, embed_sha1, self.data_download_cache).download()

                embedding_bundle = baseline.embeddings.load_embeddings(name, embed_file=embed_file, known_vocab=vocabs[name], embed_type=embed_type, **embeddings_section)

                embeddings_map[name] = embedding_bundle['embeddings']
                out_vocabs[name] = embedding_bundle['vocab']
            else:
                dsz = embeddings_section.pop('dsz')
                embedding_bundle = baseline.embeddings.load_embeddings(name, dsz=dsz, known_vocab=vocabs[name], embed_type=embed_type, **embeddings_section)
                embeddings_map[name] = embedding_bundle['embeddings']
                out_vocabs[name] = embedding_bundle['vocab']

        return embeddings_map, out_vocabs

    def get_basedir(self):
        """Return the base directory if provided, or CWD
        """
        return self.config_params.get('basedir', './{}'.format(self.task_name()))


@exporter
@register_task
class ClassifierTask(Task):

    def __init__(self, logging_config, mead_settings_config, **kwargs):
        super(ClassifierTask, self).__init__(logging_config, mead_settings_config, **kwargs)

    @classmethod
    def task_name(cls):
        return 'classify'

    def _create_backend(self):
        backend = Backend(self.config_params.get('backend', 'tf'))
        if backend.name == 'dy':
            import _dynet
            dy_params = _dynet.DynetParams()
            dy_params.from_args()
            dy_params.set_requested_gpus(1)
            if 'autobatchsz' in self.config_params['train']:
                self.config_params['train']['trainer_type'] = 'autobatch'
                dy_params.set_autobatch(True)
                batched = False
            else:
                batched = True
            dy_params.init()
            backend.params = {'pc': _dynet.ParameterCollection(), 'batched': batched}
        elif backend.name == 'tf':
            # FIXME this should be registered as well!
            from mead.tf.exporters import ClassifyTensorFlowExporter
            backend.exporter = ClassifyTensorFlowExporter

        backend.load(self.task_name())

        return backend

    def _setup_task(self):
        super(ClassifierTask, self)._setup_task()
        if self.config_params['preproc'].get('clean', False) is True:
            self.config_params['preproc']['clean_fn'] = baseline.TSVSeqLabelReader.do_clean
            print('Clean')
        else:
            self.config_params['preproc']['clean_fn'] = None

    def initialize(self, embeddings):
        embeddings = read_config_file_or_json(embeddings, 'embeddings')
        embeddings_set = index_by_label(embeddings)
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']))
        vocab, self.labels = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']], min_f=Task._get_min_f(self.config_params))
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocab, self.config_params['features'])
        baseline.save_vocabs(self.get_basedir(), self.feat2index)

    def _create_model(self):
        unif = self.config_params.get('unif', 0.1)
        model = self.config_params['model']
        model['unif'] = model.get('unif', unif)
        lengths_key = model.get('lengths_key', self.primary_key)
        if lengths_key is not None:
            if not lengths_key.endswith('_lengths'):
                lengths_key = '{}_lengths'.format(lengths_key)
            model['lengths_key'] = lengths_key
        if self.backend.params is not None:
            for k, v in self.backend.params.items():
                model[k] = v
        return baseline.model.create_model(self.embeddings, self.labels, **model)

    def _load_dataset(self):
        self.train_data = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'],
                                           shuffle=True,
                                           sort_key=self.config_params['loader'].get('sort_key'))
        self.valid_data = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'])
        self.test_data = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params.get('test_batchsz', 1))


@exporter
@register_task
class TaggerTask(Task):

    def __init__(self, logging_config, mead_settings_config, **kwargs):
        super(TaggerTask, self).__init__(logging_config, mead_settings_config, **kwargs)

    @classmethod
    def task_name(cls):
        return 'tagger'

    def _create_backend(self):
        backend = Backend(self.config_params.get('backend', 'tf'))

        if backend.name == 'pytorch':
            self.config_params['preproc']['trim'] = True
        elif backend.name == 'dy':
            import _dynet
            dy_params = _dynet.DynetParams()
            dy_params.from_args()
            dy_params.set_requested_gpus(1)
            if 'autobatchsz' in self.config_params['train']:
                dy_params.set_autobatch(True)
            else:
                raise Exception('Tagger currently only supports autobatching.'
                                'Change "batchsz" to 1 and under "train", set "autobatchsz" to your desired batchsz')
            dy_params.init()
            backend.params = {'pc': _dynet.ParameterCollection(), 'batched': False}
            self.config_params['preproc']['trim'] = True
        else:
            self.config_params['preproc']['trim'] = False
            # FIXME These should be registered instead
            from mead.tf.exporters import TaggerTensorFlowExporter
            backend.exporter = TaggerTensorFlowExporter

        backend.load(self.task_name())

        return backend

    def initialize(self, embeddings):
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']))
        embeddings = read_config_file_or_json(embeddings, 'embeddings')
        embeddings_set = index_by_label(embeddings)
        vocabs = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']], min_f=Task._get_min_f(self.config_params))
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocabs, self.config_params['features'])
        baseline.save_vocabs(self.get_basedir(), self.feat2index)

    def _create_model(self):
        labels = self.reader.label2index
        span_type = self.config_params['train'].get('span_type')
        constrain = bool(self.config_params['model'].get('constrain_decode', False))
        self.config_params['model']['span_type'] = span_type
        if span_type is not None and constrain:
            self.config_params['model']['constraint'] = self.backend.transition_mask(
                labels, span_type, Offsets.GO, Offsets.EOS, Offsets.PAD
            )

        model = self.config_params['model']
        unif = self.config_params.get('unif', 0.1)
        model['unif'] = model.get('unif', unif)

        lengths_key = model.get('lengths_key', self.primary_key)
        if lengths_key is not None:
            if not lengths_key.endswith('_lengths'):
                lengths_key = '{}_lengths'.format(lengths_key)
            model['lengths_key'] = lengths_key

        if self.backend.params is not None:
            for k, v in self.backend.params.items():
                model[k] = v
        return baseline.model.create_tagger_model(self.embeddings, labels, **self.config_params['model'])

    def _load_dataset(self):
        # TODO: get rid of sort_key=self.primary_key in favor of something explicit?
        self.train_data, _ = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'],
                                              shuffle=True,
                                              sort_key='{}_lengths'.format(self.primary_key))
        self.valid_data, _ = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'], sort_key=None)
        self.test_data, self.txts = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params.get('test_batchsz', 1), shuffle=False, sort_key=None)

    def train(self):
        self._load_dataset()
        baseline.save_vectorizers(self.get_basedir(), self.vectorizers)
        model = self._create_model()
        conll_output = self.config_params.get("conll_output", None)
        baseline.train.fit(model, self.train_data, self.valid_data, self.test_data, conll_output=conll_output, txts=self.txts, **self.config_params['train'])
        baseline.zip_files(self.get_basedir())
        self._close_reporting_hooks()
        return model


@exporter
@register_task
class EncoderDecoderTask(Task):

    def __init__(self, logging_config, mead_settings_config, **kwargs):
        super(EncoderDecoderTask, self).__init__(logging_config, mead_settings_config, **kwargs)

    @classmethod
    def task_name(cls):
        return 'seq2seq'

    def _create_backend(self):
        backend = Backend(self.config_params.get('backend', 'tf'))
        if backend.name == 'pytorch':
            from baseline.pytorch import show_examples_pytorch
            self.config_params['preproc']['show_ex'] = show_examples_pytorch
            self.config_params['preproc']['trim'] = True
        elif backend.name == 'dy':
            import _dynet
            dy_params = _dynet.DynetParams()
            dy_params.from_args()
            dy_params.set_requested_gpus(1)
            if 'autobatchsz' in self.config_params['train']:
                self.config_params['train']['trainer_type'] = 'autobatch'
                dy_params.set_autobatch(True)
                batched = False
            else:
                batched = True
            dy_params.init()
            backend.params = {'pc': _dynet.ParameterCollection(), 'batched': batched}
            from baseline.dy import show_examples_dynet
            self.config_params['preproc']['show_ex'] = show_examples_dynet
            self.config_params['preproc']['trim'] = True
        else:
            self.config_params['preproc']['trim'] = True
            # FIXME Replace with registration
            from baseline.tf import show_examples_tf
            self.config_params['preproc']['show_ex'] = show_examples_tf
            from mead.tf.exporters import Seq2SeqTensorFlowExporter
            backend.exporter = Seq2SeqTensorFlowExporter
        backend.load(self.task_name())

        return backend

    def initialize(self, embeddings):
        embeddings = read_config_file_or_json(embeddings, 'embeddings')
        embeddings_set = index_by_label(embeddings)
        self.dataset = DataDownloader(self.dataset, self.data_download_cache, True).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}\n[vocab file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file'], self.dataset.get('vocab_file',"None")))
        vocab_file = self.dataset.get('vocab_file')
        if vocab_file is not None:
            vocab1, vocab2 = self.reader.build_vocabs([vocab_file], min_f=Task._get_min_f(self.config_params))
        else:
            vocab1, vocab2 = self.reader.build_vocabs([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']], min_f=Task._get_min_f(self.config_params))

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
        baseline.save_vocabs(self.get_basedir(), self.feat2src)
        self.tgt_embeddings, self.feat2tgt = self._create_embeddings(embeddings_set, {'tgt': vocab2}, [features_tgt])
        baseline.save_vocabs(self.get_basedir(), self.feat2tgt)
        self.tgt_embeddings = self.tgt_embeddings['tgt']
        self.feat2tgt = self.feat2tgt['tgt']

    def _load_dataset(self):
        self.train_data = self.reader.load(self.dataset['train_file'],
                                           self.feat2src, self.feat2tgt,
                                           self.config_params['batchsz'],
                                           shuffle=True,
                                           sort_key='{}_lengths'.format(self.primary_key))

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
        self.config_params['model']["unif"] = self.config_params["unif"]
        model = self.config_params['model']
        unif = self.config_params.get('unif', 0.1)
        model['unif'] = model.get('unif', unif)
        lengths_key = model.get('src_lengths_key', self.primary_key)
        if lengths_key is not None:
            if not lengths_key.endswith('_lengths'):
                lengths_key = '{}_lengths'.format(lengths_key)
            model['src_lengths_key'] = lengths_key
        if self.backend.params is not None:
            for k, v in self.backend.params.items():
                model[k] = v
        return baseline.model.create_seq2seq_model(self.src_embeddings, self.tgt_embeddings, **self.config_params['model'])

    def train(self):

        num_ex = self.config_params['num_valid_to_show']

        if num_ex > 0:
            print('Showing examples')
            preproc = self.config_params['preproc']
            show_ex_fn = preproc['show_ex']
            rlut1 = baseline.revlut(self.feat2src[self.primary_key])
            rlut2 = baseline.revlut(self.feat2tgt)
            self.config_params['train']['after_train_fn'] = lambda model: show_ex_fn(model,
                                                                                     self.valid_data, rlut1, rlut2,
                                                                                     self.feat2tgt,
                                                                                     preproc['mxlen'], False, 0,
                                                                                     num_ex, reverse=False)
        super(EncoderDecoderTask, self).train()


@exporter
@register_task
class LanguageModelingTask(Task):

    def __init__(self, logging_config, mead_settings_config, **kwargs):
        super(LanguageModelingTask, self).__init__(logging_config, mead_settings_config, **kwargs)

    @classmethod
    def task_name(cls):
        return 'lm'

    def _create_task_specific_reader(self):
        self._create_vectorizers()

        reader_params = self.config_params['loader']
        reader_params['nctx'] = reader_params.get('nctx', self.config_params.get('nctx', self.config_params.get('nbptt', 35)))
        reader_params['clean_fn'] = reader_params.get('clean_fn', self.config_params['preproc'].get('clean_fn'))
        reader_params['mxlen'] = self.vectorizers[self.primary_key].mxlen
        return baseline.reader.create_reader(self.task_name(), self.vectorizers, self.config_params['preproc'].get('trim', False), **reader_params)

    def _create_backend(self):
        backend = Backend(self.config_params.get('backend', 'tf'))

        if backend.name == 'pytorch':
            self.config_params['preproc']['trim'] = True

        elif backend.name == 'dy':
            self.config_params['preproc']['trim'] = True
            import _dynet
            dy_params = _dynet.DynetParams()
            dy_params.from_args()
            dy_params.set_requested_gpus(1)
            if 'autobatchsz' in self.config_params['train']:
                dy_params.set_autobatch(True)
                batched = False
            else:
                batched = True
            dy_params.init()
            backend.params = {'pc': _dynet.ParameterCollection(), 'batched': batched}

        backend.load(self.task_name())
        return backend

    def initialize(self, embeddings):
        embeddings = read_config_file_or_json(embeddings, 'embeddings')
        embeddings_set = index_by_label(embeddings)
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print("[train file]: {}\n[valid file]: {}\n[test file]: {}".format(self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']))
        vocabs = self.reader.build_vocab([self.dataset['train_file'], self.dataset['valid_file'], self.dataset['test_file']], min_f=Task._get_min_f(self.config_params))
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocabs, self.config_params['features'])
        baseline.save_vocabs(self.get_basedir(), self.feat2index)

    def _load_dataset(self):
        tgt_key = self.config_params['loader'].get('tgt_key', self.primary_key)
        self.train_data = self.reader.load(self.dataset['train_file'], self.feat2index, self.config_params['batchsz'], tgt_key=tgt_key)
        self.valid_data = self.reader.load(self.dataset['valid_file'], self.feat2index, self.config_params['batchsz'], tgt_key=tgt_key)
        self.test_data = self.reader.load(self.dataset['test_file'], self.feat2index, self.config_params['batchsz'], tgt_key=tgt_key)

    def _create_model(self):

        model = self.config_params['model']
        unif = self.config_params.get('unif', 0.1)
        model['unif'] = model.get('unif', unif)
        model['batchsz'] = self.config_params['batchsz']
        model['tgt_key'] = self.config_params['loader'].get('tgt_key', self.primary_key)
        if self.backend.params is not None:
            for k, v in self.backend.params.items():
                model[k] = v
        return baseline.model.create_lang_model(self.embeddings, **model)

    def train(self):
        self._load_dataset()
        if self.config_params['train'].get('lr_scheduler_type', None) == 'zaremba':
            first_range = int(self.config_params['train']['start_decay_epoch'] * self.train_data.steps)
            self.config_params['train']['bounds'] = [first_range] + list(
                np.arange(
                    self.config_params['train']['start_decay_epoch'] + 1,
                    self.config_params['train']['epochs'] + 1,
                    dtype=np.int32
                ) * self.train_data.steps
            )
        baseline.save_vectorizers(self.get_basedir(), self.vectorizers)
        model = self._create_model()
        baseline.train.fit(model, self.train_data, self.valid_data, self.test_data, **self.config_params['train'])
        baseline.zip_files(self.get_basedir())
        self._close_reporting_hooks()


    @staticmethod
    def _num_steps_per_epoch(num_examples, nctx, batchsz):
        rest = num_examples // batchsz
        return rest // nctx
