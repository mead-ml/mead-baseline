"""Run any MEAD embeddings within the SentEval framework

The SentEval framework (https://github.com/facebookresearch/SentEval) facilitates
testing the quality of sentence embeddings.  To prepare your data, you can clone
that repo and do a `pip install -e .`.  This program allows you to control which
sets of tasks are run.  By default, it will run all STS and classification probing
examples.

Please note that its common when evaluating against STS to use a different approach
from the one in SentEval (following SentenceBERT).  A drop-in replacement is available
from the SimCSE repository (https://github.com/princeton-nlp/SimCSE).

To use their modified benchmark, clone that repo instead
and within their SentEval directory, do a `pip install -e .`, and pay attention instead
to the ALL Spearman metrics:

"""
import argparse
import baseline
import sys
from mead.api_examples.preproc_utils import *
from baseline.embeddings import load_embeddings_overlay
from eight_mile.utils import read_config_stream
from baseline.pytorch.embeddings import *
from baseline.embeddings import *
from baseline.vectorizers import *
from mead.utils import convert_path, parse_extra_args, configure_logger, parse_and_merge_overrides, read_config_file_or_json, index_by_label
import senteval
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
DEFAULT_SETTINGS_LOC = 'config/mead-settings.json'
DEFAULT_DATASETS_LOC = 'config/datasets.json'
DEFAULT_EMBEDDINGS_LOC = 'config/embeddings.json'
DEFAULT_VECTORIZERS_LOC = 'config/vecs.json'
SUBWORD_EXTRA = 30

def main():
    parser = argparse.ArgumentParser(description='Run senteval harness')
    parser.add_argument('--nctx', default=512, type=int)
    parser.add_argument("--module", default=None, help="Module containing custom tokenizers")
    parser.add_argument('--tasks', nargs="+", default=['sts', 'class', 'probe'])
    parser.add_argument('--batchsz', default=20, type=int)
    parser.add_argument('--tok', help='Optional tokenizer, e.g. "gpt2" or "basic". These can be defined in extra module')
    parser.add_argument('--pool', help='Should a reduction be applied on the embeddings?  Only use if your embeddings arent already pooled', type=str)
    parser.add_argument('--vec_id', help='Reference to a specific embedding type')
    parser.add_argument('--embed_id', help='What type of embeddings to use')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument('--max_len1d', type=int, default=100)
    parser.add_argument('--embeddings', help='index of embeddings: local file, remote URL or mead-ml/hub ref', type=convert_path)
    parser.add_argument('--vecs', help='index of vectorizers: local file, remote URL or hub mead-ml/ref', type=convert_path)
    parser.add_argument('--fast', help="Run fast, but not necessarily as accurate", action='store_true')
    parser.add_argument('--data', help="Path to senteval data",
                        default=os.path.expanduser("~/dev/work/SentEval/data"))
    args = parser.parse_args()

    if args.module:
        logger.warning("Loading custom user module %s for masking rules and tokenizers", args.module)
        baseline.import_user_module(args.module)


    tokenizer = create_tokenizer(args.tok) if args.tok else None

    args.embeddings = convert_path(DEFAULT_EMBEDDINGS_LOC) if args.embeddings is None else args.embeddings
    args.embeddings = read_config_stream(args.embeddings)

    args.vecs = convert_path(DEFAULT_VECTORIZERS_LOC) if args.vecs is None else args.vecs

    vecs_index = read_config_stream(args.vecs)
    vecs_set = index_by_label(vecs_index)
    vec_params = vecs_set[args.vec_id]
    vec_params['mxlen'] = args.nctx

    if 'transform' in vec_params:
        vec_params['transform_fn'] = vec_params['transform']

    if 'transform_fn' in vec_params and isinstance(vec_params['transform_fn'], str):
        vec_params['transform_fn'] = eval(vec_params['transform_fn'])

    vectorizer = create_vectorizer(**vec_params)
    if not isinstance(vectorizer, HasPredefinedVocab):
        raise Exception("We currently require a vectorizer with a pre-defined vocab to run this script")
    embeddings_index = read_config_stream(args.embeddings)
    embeddings_set = index_by_label(embeddings_index)
    embeddings_params = embeddings_set[args.embed_id]
    embeddings = load_embeddings_overlay(embeddings_set, embeddings_params, vectorizer.vocab)

    embedder = embeddings['embeddings']
    embedder.to(args.device).eval()

    def _mean_pool(inputs, embeddings):
        mask = (inputs != 0)
        seq_lengths = mask.sum(1).unsqueeze(-1)
        return embeddings.sum(1)/seq_lengths

    def _zero_tok_pool(_, embeddings):
        pooled = embeddings[:, 0]
        return pooled

    def _max_pool(inputs, embeddings):
        mask = (inputs != 0)
        embeddings = embeddings.masked_fill(mask.unsqueeze(-1) == False, -1e8)
        return torch.max(embeddings, 1, False)[0]

    if args.pool:
        if args.pool == 'max':
            pool = _max_pool
        elif args.pool == 'zero' or args.pool == 'cls':
            pool = _zero_tok_pool
        else:
            pool = _mean_pool
    else:
        pool = lambda x, y: y

    params_senteval = {'task_path': args.data, 'usepytorch': True, 'kfold': 10}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                     'tenacity': 5, 'epoch_size': 4}
    if args.fast:
        logging.info("Setting fast params")
        params_senteval['kfold'] = 5
        params_senteval['classifier']['epoch_size'] = 2
        params_senteval['classifier']['tenacity'] = 3
        params_senteval['classifier']['batch_size'] = 128

    # SentEval prepare and batcher
    def prepare(params, samples):
        max_sample = max(len(s) for s in samples)
        vectorizer.mxlen = min(args.nctx, max_sample + SUBWORD_EXTRA)
        logging.info('num_samples %d, mxlen set to %d', max_sample, vectorizer.mxlen)

    def batcher(params, batch):
        if not tokenizer:
            batch = [sent if sent != [] else ['.'] for sent in batch]
        else:
            batch = [tokenizer(' '.join(sent)) for sent in batch]

        vs = []
        for sent in batch:
            v, l = vectorizer.run(sent, vectorizer.vocab)
            vs.append(v)
        vs = np.stack(vs)
        with torch.no_grad():
            inputs = torch.tensor(vs, device=args.device)
            encoding = embedder(inputs)
            encoding = pool(inputs, encoding)
            encoding = encoding.cpu().numpy()
        return encoding

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = []
    if 'sts' in args.tasks:
        transfer_tasks += ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICKRelatedness', 'STSBenchmark']
    if 'class' in args.tasks:
        transfer_tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                           'SICKEntailment']
    if 'probe' in args.tasks:
        transfer_tasks += ['Length', 'WordContent', 'Depth', 'TopConstituents',
                           'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                           'OddManOut', 'CoordinationInversion']

    results = se.eval(transfer_tasks)
    print(results)


if __name__ == '__main__':
    main()
