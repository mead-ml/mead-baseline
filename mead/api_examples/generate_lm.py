import numpy as np
import torch
import logging
import os
import glob
from argparse import ArgumentParser
import baseline
#from eight_mile.pytorch.layers import EmbeddingsStack
from eight_mile.pytorch.serialize import tlm_load_state_dict, load_tlm_npz
from baseline.pytorch.lm import TransformerLanguageModel
from eight_mile.utils import str2bool, read_json, Offsets, revlut
from baseline.vectorizers import Token1DVectorizer, BPEVectorizer1D
from baseline.pytorch.embeddings import *
from eight_mile.pytorch.layers import find_latest_checkpoint
logger = logging.getLogger(__file__)


def decode_sentence(model, vectorizer, query, word2index, index2word, device, end_token='<EOS>', sample=True, sample_temperature=1.0):
    vec, length = vectorizer.run(query, word2index)
    bpe = [index2word[v] for v in vec if v != Offsets.PAD]
    logger.info('[BPE] ' + ' '.join(bpe))
    toks = torch.from_numpy(vec).to(device=device)

    with torch.no_grad():

        words = []
        for i in range(100):
            predictions, _ = model({'x': toks.unsqueeze(0)}, None)
            predictions = predictions.squeeze(0)

            if not sample:
                output = torch.argmax(predictions[length + i - 1], -1).item()
                word = index2word[output]
            else:
                sample_dist = torch.softmax(predictions[length + i - 1] / sample_temperature, -1)
                output = torch.multinomial(sample_dist, num_samples=1)
                output = output.squeeze(0).item()
                word = index2word[output]
            words.append(word)
            toks[length + i] = output
            if word == end_token:
                break

        return words[:-1]


def create_model(embeddings, d_model, d_ff, num_heads, num_layers, rpr_k, rpr_value_on, d_k, checkpoint_name, activation):
    rpr_k = listify(rpr_k)

    if len(rpr_k) == 0 or rpr_k[0] < 1:
        rpr_k = None
    elif len(rpr_k) == 1:
        rpr_k = rpr_k[0]

    logger.info("Creating tied encoder decoder model")
    model = TransformerLanguageModel.create({'x': embeddings},
                                            hsz=d_model,
                                            d_ff=d_ff,
                                            tie_weights=True,
                                            dropout=0,
                                            gpu=False,
                                            num_heads=num_heads,
                                            layers=num_layers,
                                            rpr_k=rpr_k,
                                            rpr_value_on=rpr_value_on,
                                            d_k=d_k,
                                            activation=activation,
                                            src_keys=['x'], tgt_key='x')
    if checkpoint_name.endswith('npz'):
        load_tlm_npz(model, checkpoint_name)
    else:
        tlm_load_state_dict(model, checkpoint_name)
    model.eval()
    print(model)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--checkpoint", type=str, help='Checkpoint name or directory to load')
    parser.add_argument("--sample", type=str2bool, help='Sample from the decoder?  Defaults to `true`', default=True)
    parser.add_argument("--query", type=str, default='hello , <unk> are you today ?')
    parser.add_argument("--dataset_cache", type=str, default=os.path.expanduser('~/.bl-data'),
                        help="Path or url of the dataset cache")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--nctx", type=int, default=128, help="Max context length (for both encoder and decoder)")
    parser.add_argument("--embed_type", type=str, default='default',
                        help="register label of the embeddings, so far support positional or learned-positional")
    parser.add_argument("--subword_model_file", type=str, required=True)
    parser.add_argument("--subword_vocab_file", type=str, required=True)
    parser.add_argument('--go_token', default='<GO>')
    parser.add_argument('--end_token', default='<EOU>')
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument('--rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')
    parser.add_argument("--rpr_value_on", help="Use different embeddings for RPV key and value", type=str2bool, default=False)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument('--temperature', help='Sample temperature during generation', default=1.0, type=float)

    args = parser.parse_args()
    if args.sample:
        logger.info("Sampling with temperature %f", args.temperature)
    else:
        logger.info("Sampling is turned off")
    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
        args.device = torch.device("cuda", 0)


    if os.path.isdir(args.checkpoint):
        checkpoint, _ = find_latest_checkpoint(args.checkpoint)
        logger.warning("Found latest checkpoint %s", checkpoint)
    else:
        checkpoint = args.checkpoint

    vectorizer = BPEVectorizer1D(model_file=args.subword_model_file, vocab_file=args.subword_vocab_file, mxlen=args.nctx, emit_begin_tok=args.go_token)
    vocab = vectorizer.vocab.copy()
    # If we are not using chars, then use 'x' for both input and output
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, counts=False, known_vocab=vocab, embed_type=args.embed_type, preserve_vocab_indices=True)
    embeddings = preproc_data['embeddings']
    vocab = preproc_data['vocab']
    model = create_model(embeddings, d_model=args.d_model, d_ff=args.d_ff, num_heads=args.num_heads, num_layers=args.num_layers,
                         rpr_k=args.rpr_k, rpr_value_on=args.rpr_value_on, d_k=args.d_k, checkpoint_name=checkpoint, activation=args.activation)
    model.to(args.device)

    index2word = revlut(vocab)
    print('[Query]', args.query)
    bpe_out = decode_sentence(model, vectorizer, args.query.split(), vocab, index2word, args.device,
                              end_token=args.end_token, sample=args.sample, sample_temperature=args.temperature)
    unbpe = ' '.join(bpe_out).replace('@@ ', '')
    print('[Response]', unbpe)

if __name__ == '__main__':
    main()
