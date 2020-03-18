import numpy as np
import torch
import logging
import os
import glob
from argparse import ArgumentParser
import baseline
from transformer_utils import TiedEmbeddingsSeq2SeqModel
from baseline.utils import str2bool, read_json, Offsets, revlut
from baseline.vectorizers import Token1DVectorizer, BPEVectorizer1D

logger = logging.getLogger(__file__)


def decode_sentence(model, vectorizer, query, word2index, index2word, device, max_response_length, sample=True):
    vec, length = vectorizer.run(query, word2index)
    toks = torch.from_numpy(vec).unsqueeze(0).to(device=device)
    length = torch.from_numpy(np.array(length)).unsqueeze(0).to(device=device)
    EOU = word2index.get('<EOU>')
    response = []
    with torch.no_grad():
        dst = [Offsets.GO]
        for i in range(max_response_length):
            dst_tensor = torch.from_numpy(np.array(dst)).unsqueeze(0).to(device=device)
            predictions = model({'x': toks, 'src_len': length, 'dst': dst_tensor})

            if not sample:
                output = torch.argmax(predictions, -1).squeeze(0)
                output = output[-1].item()
            else:
                # using a multinomial distribution to predict the word returned by the model
                predictions = predictions.exp().squeeze(0)
                output = torch.multinomial(predictions, num_samples=1).squeeze(0)[-1].item()


            dst.append(output)
            response.append(index2word.get(dst[-1], '<ERROR>'))
            if output == Offsets.EOS or output == EOU:
                break
    return response


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    step_num = 0
    for f in glob.glob(os.path.join(checkpoint_dir, "checkpoint*")):
        this_step_num = int(f.split("-")[-1])
        if this_step_num > step_num:
            checkpoint = f
            step_num = this_step_num
    logger.warning("Found latest checkpoint %s", checkpoint)

    return checkpoint


def create_model(embeddings, d_model, d_ff, num_heads, num_layers, rpr_k, d_k, checkpoint_name):
    if len(rpr_k) == 0 or rpr_k[0] < 1:
        rpr_k = None
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
           "rpr_k": rpr_k}
    model = TiedEmbeddingsSeq2SeqModel(embeddings, **hps)
    model.load_state_dict(torch.load(checkpoint_name))
    model.eval()
    print(model)
    return model


def run():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--checkpoint", type=str, help='Checkpoint name or directory to load')
    parser.add_argument("--sample", type=str2bool, help='Sample from the decoder?  Defaults to `true`', default=1)
    parser.add_argument("--vocab", type=str, help='Vocab file to load', required=False)
    parser.add_argument("--query", type=str, default='hello how are you ?')
    parser.add_argument("--dataset_cache", type=str, default=os.path.expanduser('~/.bl-data'),
                        help="Path or url of the dataset cache")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--nctx", type=int, default=64, help="Max context length (for both encoder and decoder)")
    parser.add_argument("--embed_type", type=str, default='learned-positional',
                        help="register label of the embeddings, so far support positional or learned-positional")
    parser.add_argument("--subword_model_file", type=str, required=True)
    parser.add_argument("--subword_vocab_file", type=str, required=True)
    parser.add_argument('--rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[3, 5, 48, 48, 48, 48], nargs='+')

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
        args.device = torch.device("cuda", 0)


    vocab_file = args.vocab

    if os.path.isdir(args.checkpoint):
        vocab_file = os.path.join(args.checkpoint, 'vocabs.json')
        checkpoint = find_latest_checkpoint(args.checkpoint)
    else:
        checkpoint = args.checkpoint

    vocab = read_json(vocab_file)
    # If we are not using chars, then use 'x' for both input and output
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, known_vocab=vocab, embed_type=args.embed_type)
    embeddings = preproc_data['embeddings']

    model = create_model(embeddings, d_model=args.d_model, d_ff=args.d_ff, num_heads=args.num_heads, num_layers=args.num_layers,
                         rpr_k=args.rpr_k, d_k=args.d_k, checkpoint_name=checkpoint)
    model.to(args.device)

    vectorizer = BPEVectorizer1D(model_file=args.subword_model_file, vocab_file=args.subword_vocab_file, mxlen=args.nctx)
    index2word = revlut(vocab)
    print('[Query]', args.query)
    print('[Response]', ' '.join(decode_sentence(model, vectorizer, args.query.split(), vocab, index2word, args.device, max_response_length=args.nctx, sample=args.sample)))

run()
