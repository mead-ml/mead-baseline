import argparse
import baseline
import sys
from baseline.embeddings import load_embeddings_overlay
from eight_mile.utils import read_config_stream
from baseline.pytorch.embeddings import *
from baseline.embeddings import *
from baseline.vectorizers import *
from mead.api_examples.preproc_utils import create_tokenizer
from mead.utils import convert_path, parse_extra_args, configure_logger, parse_and_merge_overrides, read_config_file_or_json, index_by_label
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

DEFAULT_SETTINGS_LOC = 'config/mead-settings.json'
DEFAULT_DATASETS_LOC = 'config/datasets.json'
DEFAULT_EMBEDDINGS_LOC = 'config/embeddings.json'
DEFAULT_VECTORIZERS_LOC = 'config/vecs.json'

def main():
    parser = argparse.ArgumentParser(description='Encode a sentence as an embedding')
    parser.add_argument('--subword_model_file', help='Subword model file')
    parser.add_argument('--nctx', default=256, type=int)
    parser.add_argument('--batchsz', default=20, type=int)
    parser.add_argument('--vec_id', default='bert-base-uncased', help='Reference to a specific embedding type')
    parser.add_argument('--embed_id', default='bert-base-uncased', help='What type of embeddings to use')
    parser.add_argument('--file', required=True)
    parser.add_argument('--column', type=str)
    parser.add_argument('--output', default='embeddings.npz')
    parser.add_argument('--pool', help='Should a reduction be applied on the embeddings?  Only use if your embeddings arent already pooled', type=str)
    parser.add_argument('--embeddings', help='index of embeddings: local file, remote URL or mead-ml/hub ref', type=convert_path)
    parser.add_argument('--vecs', help='index of vectorizers: local file, remote URL or hub mead-ml/ref', type=convert_path)
    parser.add_argument('--cuda', type=baseline.str2bool, default=True)
    parser.add_argument('--has_header', action="store_true")
    parser.add_argument("--tokenizer_type", type=str, help="Optional tokenizer, default is to use string split")
    parser.add_argument('--faiss_index', help="If provided, we will build a FAISS index and store it here")
    parser.add_argument('--quoting', default=3, help='0=QUOTE_MINIMAL 1=QUOTE_ALL 2=QUOTE_NONNUMERIC 3=QUOTE_NONE', type=int)
    parser.add_argument('--sep', default='\t')

    args = parser.parse_args()

    if not args.has_header:
        if not args.column:
            args.column = 0
        column = int(args.column)
    else:
        column = args.column

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
    tokenizer = create_tokenizer(args.tokenizer_type)
    vectorizer = create_vectorizer(**vec_params)
    if not isinstance(vectorizer, HasPredefinedVocab):
        raise Exception("We currently require a vectorizer with a pre-defined vocab to run this script")
    embeddings_index = read_config_stream(args.embeddings)
    embeddings_set = index_by_label(embeddings_index)
    embeddings_params = embeddings_set[args.embed_id]
    # If they dont want CUDA try and get the embedding loader to use CPU
    embeddings_params['cpu_placement'] = not args.cuda
    embeddings = load_embeddings_overlay(embeddings_set, embeddings_params, vectorizer.vocab)

    vocabs = {'x': embeddings['vocab']}
    embedder = embeddings['embeddings'].cpu()
    embedder.eval()
    if args.cuda:
        embedder = embedder.cuda()


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

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    df = pd.read_csv(args.file, header='infer' if args.has_header else None, sep=args.sep)
    col = df[column]
    batches = []
    as_list = col.tolist()
    num_batches = math.ceil(len(as_list) / args.batchsz)
    pg = baseline.create_progress_bar(num_batches, name='tqdm')
    embed_dsz = 0
    for i, batch in enumerate(chunks(as_list, args.batchsz)):
        pg.update()
        with torch.no_grad():
            vecs = []
            for line in batch:
                tokenized = tokenizer(line)
                vec, l = vectorizer.run(tokenized, vocabs['x'])
                vecs.append(vec)
            vecs = torch.tensor(np.stack(vecs))
            if args.cuda:
                vecs = vecs.cuda()
            embedding = embedder(vecs)
            pooled_batch = pool(vecs, embedding).cpu().numpy()
            batches += [x for x in pooled_batch]

    np.savez(args.output, embeddings=batches, text=as_list)
    if args.faiss_index:
        import faiss
        index = faiss.IndexFlatIP(batches[0].shape[-1])
        batches = np.stack(batches)
        faiss.normalize_L2(batches)
        index.add(batches)
        faiss.write_index(index, args.faiss_index)


if __name__ == '__main__':
    main()
