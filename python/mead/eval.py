from copy import deepcopy
import argparse
import logging
import baseline
from baseline.reporting import create_reporting
from baseline.reader import create_reader
from baseline.train import create_trainer
from baseline.utils import read_config_stream, str2bool, import_user_module
from baseline.services import ClassifierService, TaggerService, EncoderDecoderService, LanguageModelService
from mead.utils import configure_logger, convert_path, parse_extra_args
from mead.tasks import merge_reporting_with_settings, Backend


DEFAULT_SETTINGS_LOC = 'config/mead-settings.json'
DEFAULT_LOGGING_LOC = 'config/logging.json'
logger = logging.getLogger('mead')
SERVICES = {
    'classify': ClassifierService,
    'tagger': TaggerService,
    'seq2seq': EncoderDecoderService,
    'lm': LanguageModelService
}


def feature_index_mapping(features):
    if not features:
        return {}
    elif ':' in features[0]:
        return {feature.split(':')[0]: int(feature.split(':')[1]) for feature in features}
    else:
        return {feature: index for index, feature in enumerate(features)}


def get_service(task, services=SERVICES):
    return services[task]


def get_vectorizers(task, model):
    vectorizers = model.vectorizers
    if task == 'seq2seq':
        vectorizers = deepcopy(model.src_vectorizers)
        vectorizers['tgt'] = deepcopy(model.tgt_vectorizer)
    return vectorizers


def patch_reader(task, model, reader):
    if task == 'classify':
        reader.label2index = {l: i for i, l in enumerate(model.model.labels)}
    if task == 'tagger':
        reader.label2index = model.model.labels
        reader.label_vectorizer.mxlen = model.vectorizers[list(model.vectorizers.keys())[0]].mxlen
    if task == 'seq2seq':
        # This might be skippable when PR #318 is merged
        reader.tgt_vectorizer.vectorizer.mxlen = model.tgt_vectorizer.mxlen
    return reader


def load_data(task, reader, model, dataset, batchsz):
    if task == 'seq2seq':
        data = reader.load(dataset, model.src_vocabs, model.tgt_vocab, batchsz)
    elif task == 'lm':
        data = reader.load(dataset, model.vocabs, batchsz, tgt_key=model.model.tgt_key)
    elif task == 'classify':
        if hasattr(reader, 'load_text'):
            data = reader.load_text(dataset, model.vocabs, batchsz)
        else:
            data = reader.load(dataset, model.vocabs, batchsz)
    else:
        data = reader.load(dataset, model.vocabs, batchsz)
    if not isinstance(data, tuple):
        data = (data, )
    return data


def process_reader_options(reader_options):
    if 'clean_fn' in reader_options:
        reader_options['clean_fn'] = eval(reader_options['clean_fn'])
    return reader_options


def get_trainer(model, trainer, verbose, backend, **kwargs):
    if backend == 'tf':
        with model.model.sess.graph.as_default():
            trainer = create_trainer(
                model.model,
                type=trainer,
                verbose=verbose,
                eval_mode=True,
                basedir='',
                **kwargs
            )
    else:
        trainer = create_trainer(
            model.model, type=trainer, verbose=verbose, eval_mode=True, basedir='', **kwargs
        )
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Evaluate on a dataset')
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--settings', default=DEFAULT_SETTINGS_LOC, type=convert_path)
    parser.add_argument('--modules', nargs="+", default=[])
    parser.add_argument('--reporting', nargs="+")
    parser.add_argument('--logging', default=DEFAULT_LOGGING_LOC, type=convert_path)
    parser.add_argument('--task', default='classify', choices={'classify', 'tagger', 'seq2seq', 'lm'})
    parser.add_argument('--backend', default='tf')
    parser.add_argument('--reader', default='default')
    parser.add_argument('--trim', default=True, type=str2bool)
    parser.add_argument('--batchsz', default=50)
    parser.add_argument('--trainer', default='default')
    parser.add_argument('--output', default=None)
    parser.add_argument('--remote')
    parser.add_argument(
        '--features',
        help='(optional) features in the format feature_name:index (column # in conll) or '
        'just feature names (assumed sequential)',
        default=[],
        nargs='+',
    )
    parser.add_argument('--device', default='cpu')
    # our parse_extra_args doesn't handle lists :/
    parser.add_argument('--pair_suffix', nargs='+', default=[])
    args, extra_args = parser.parse_known_args()

    args.batchsz = args.batchsz if args.task != 'lm' else 1

    named_fields = {str(v): k for k, v in feature_index_mapping(args.features).items()}

    reader_options = parse_extra_args(['reader'], extra_args)['reader']
    reader_options = process_reader_options(reader_options)
    verbose_options = parse_extra_args(['verbose'], extra_args)['verbose']
    trainer_options = parse_extra_args(['trainer'], extra_args)['trainer']
    if 'span_type' not in trainer_options:
        trainer_options['span_type'] = 'iobes'
    model_options = parse_extra_args(['model'], extra_args)['model']

    args.logging = read_config_stream(args.logging)
    configure_logger(args.logging)

    try:
        args.settings = read_config_stream(args.settings)
    except:
        logger.warning('Warning: no mead-settings file was found at [{}]'.format(args.settings))
        args.settings = {}

    backend = Backend(args.backend)
    backend.load(args.task)
    for module in args.modules:
        import_user_module(module)

    reporting = parse_extra_args(args.reporting if args.reporting is not None else [], extra_args)
    reporting_hooks, reporting = merge_reporting_with_settings(reporting, args.settings)
    reporting_fns = [x.step for x in create_reporting(reporting_hooks, reporting, {'task': args.task})]

    service = get_service(args.task)
    model = service.load(args.model, backend=args.backend, remote=args.remote, device=args.device, **model_options)

    vectorizers = get_vectorizers(args.task, model)

    reader = create_reader(
        args.task,
        vectorizers,
        args.trim,
        type=args.reader,
        named_fields=named_fields,
        pair_suffix=args.pair_suffix,
        **reader_options
    )
    reader = patch_reader(args.task, model, reader)

    data, txts = load_data(args.task, reader, model, args.dataset, args.batchsz)

    if args.task == 'seq2seq':
        trainer_options['tgt_rlut'] = {v: k for k, v in model.tgt_vocab.items()}

    trainer = get_trainer(
        model,
        args.trainer,
        verbose_options,
        backend.name,
        gpu=args.device != 'cpu',
        nogpu=args.device == 'cpu',
        **trainer_options
    )
    if args.task == 'classify':
        _ = trainer.test(data, reporting_fns=reporting_fns, phase='Test', verbose=verbose_options,
                         output=args.output, txts=txts, **model_options)
    elif args.task == 'tagger':
        _ = trainer.test(data, reporting_fns=reporting_fns, phase='Test', verbose=verbose_options,
                         conll_output=args.output, txts=txts, **model_options)
    else:
        _ = trainer.test(data, reporting_fns=reporting_fns, phase='Test', verbose=verbose_options, **model_options)


if __name__ == "__main__":
    main()
