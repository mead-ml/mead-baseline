import numpy as np
import torch
import logging
from eight_mile.utils import listify
import os
import glob
from argparse import ArgumentParser
import baseline
from eight_mile.pytorch.layers import find_latest_checkpoint
from baseline.pytorch.embeddings import *
from baseline.pytorch.seq2seq.model import TiedEmbeddingsSeq2SeqModel
from eight_mile.pytorch.serialize import load_transformer_seq2seq_npz
from eight_mile.utils import str2bool, read_json, Offsets, revlut
from baseline.vectorizers import Token1DVectorizer, BPEVectorizer1D

logger = logging.getLogger(__file__)


def decode_sentences(model, vectorizer, queries, word2index, index2word, beamsz):

    vecs = []
    lengths = []
    for query in queries:
        vec, length = vectorizer.run(query, word2index)
        vecs.append(vec)
        lengths.append(length)
    vecs = np.stack(vecs)
    lengths = np.stack(lengths)
    # B x K x T
    with torch.no_grad():
        response, _ = model.predict({'x': vecs, 'x_lengths': lengths}, beam=beamsz)
    sentences = []
    for candidate in response:
        best_sentence_idx = candidate[0]
        best_sentence = ' '.join([index2word[x] for x in best_sentence_idx if x not in [Offsets.EOS, Offsets.PAD]])
        sentences.append(best_sentence.replace('@@ ', ''))
    return sentences

def create_model(embeddings, d_model, d_ff, num_heads, num_layers, rpr_k, d_k, activation, checkpoint_name, device):
    if len(rpr_k) == 0 or rpr_k[0] < 1:
        rpr_k = [None]
    else:
        rpr_k = listify(rpr_k)
    logger.info("Creating tied encoder decoder model")
    hps = {"dsz": d_model,
           "hsz": d_model,
           "d_ff": d_ff,
           "dropout": 0.0,
           "num_heads": num_heads,
           "layers": num_layers,
           "encoder_type": "transformer",
           "decoder_type": "transformer",
           "src_lengths_key": "x_lengths",
           "d_k": d_k,
           "activation": activation,
           "rpr_k": rpr_k}
    model = TiedEmbeddingsSeq2SeqModel({'x': embeddings}, None, **hps)
    if checkpoint_name.endswith('npz'):
        load_transformer_seq2seq_npz(model, checkpoint_name)
    else:
        model.load_state_dict(torch.load(checkpoint_name, map_location=torch.device(device)))
    print(model)
    return model


def run():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--checkpoint", type=str, help='Checkpoint name or directory to load')
    parser.add_argument("--sample", type=str2bool, help='Sample from the decoder?  Defaults to `false`', default=0)
    parser.add_argument("--vocab", type=str, help='Vocab file to load', required=False)
    parser.add_argument("--input", type=str, default='hello how are you ?')
    parser.add_argument("--dataset_cache", type=str, default=os.path.expanduser('~/.bl-data'),
                        help="Path or url of the dataset cache")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--nctx", type=int, default=256, help="Max context length (for both encoder and decoder)")
    parser.add_argument("--embed_type", type=str, default='default',
                        help="register label of the embeddings, so far support positional or learned-positional")
    parser.add_argument("--subword_model_file", type=str, required=True)
    parser.add_argument("--subword_vocab_file", type=str, required=True)
    parser.add_argument("--batchsz", help="Size of a batch to pass at once", default=4, type=int)
    parser.add_argument("--beamsz", help="Size of beam to use", default=4, type=int)
    parser.add_argument("--activation", type=str, default='relu')
    parser.add_argument('--rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8]*8, nargs='+')
    #parser.add_argument("--go_token", default="<GO>")
    parser.add_argument("--end_token", default="<EOS>")
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--show_query", type=str2bool, default=False, help="Show the original query as well")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--extra_tokens", help="What extra tokens should we use", nargs="+", default=["[CLS]", "[MASK]"])
    args = parser.parse_args()

    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
        args.device = torch.device("cuda", 0)

    if os.path.isdir(args.checkpoint):
        checkpoint, _ = find_latest_checkpoint(args.checkpoint)
        logger.warning("Found latest checkpoint %s", checkpoint)
    else:
        checkpoint = args.checkpoint

    vectorizer = BPEVectorizer1D(model_file=args.subword_model_file, vocab_file=args.subword_vocab_file,
                                 mxlen=args.nctx, emit_end_tok=args.end_token, extra_tokens=args.extra_tokens)
    vocab = vectorizer.vocab
    # If we are not using chars, then use 'x' for both input and output
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, counts=False, known_vocab=vocab, embed_type=args.embed_type)
    embeddings = preproc_data['embeddings']
    vocab = preproc_data['vocab']
    model = create_model(embeddings, d_model=args.d_model, d_ff=args.d_ff, num_heads=args.num_heads, num_layers=args.num_layers,
                         rpr_k=args.rpr_k, d_k=args.d_k, checkpoint_name=checkpoint, activation=args.activation,
                         device=args.device)
    model.to(args.device)

    index2word = revlut(vocab)
    wf = None
    if args.output_file:
        wf = open(args.output_file, "w")


    batches = []
    if os.path.exists(args.input) and os.path.isfile(args.input):
        with open(args.input, 'rt', encoding='utf-8') as f:
            batch = []
            for line in f:
                text = line.strip().split()
                if len(batch) == args.batchsz:
                    batches.append(batch)
                    batch = []
                batch.append(text)

            if len(batch) > 0:
                batches.append(batch)

    else:
        batch = [args.input.split()]
        batches.append(batch)

    for queries in batches:

        outputs = decode_sentences(model, vectorizer, queries, vocab, index2word, args.beamsz)

        if args.show_query:
            for query, output in zip(queries, outputs):
                print(f"[Query] {query}")
                print(f"[Response] {output}")
        elif wf:
            for query, output in zip(queries, outputs):
                wf.write(f'{output}\n')
                wf.flush()
        else:
            for query, output in zip(queries, outputs):
                print(output)
    if wf:
        wf.close()
run()
