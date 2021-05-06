import numpy as np
import torch
import logging
from eight_mile.utils import listify
import os
import glob
from argparse import ArgumentParser
import baseline
from transformer_utils import find_latest_checkpoint
from baseline.pytorch.seq2seq.model import TiedEmbeddingsSeq2SeqModel
from eight_mile.pytorch.serialize import load_transformer_seq2seq_npz
from eight_mile.utils import str2bool, read_json, Offsets, revlut
from baseline.vectorizers import Token1DVectorizer, BPEVectorizer1D

logger = logging.getLogger(__file__)


def decode_sentence(model, vectorizer, query, word2index, index2word, device, max_response_length, sou_token, eou_token, sample=True):
    UNK = word2index.get('<UNK>')
    MASK = word2index.get('[MASK]')
    GO = word2index.get(sou_token)
    vec, length = vectorizer.run(query, word2index)

    for i in range(length):
        if vec[i] == UNK:
            vec[i] = MASK

    toks = torch.from_numpy(vec).unsqueeze(0).to(device=device)
    length = torch.from_numpy(np.array(length)).unsqueeze(0).to(device=device)
    EOU = word2index.get(eou_token)
    response = []
    with torch.no_grad():
        dst = [GO]
        for i in range(max_response_length):
            dst_tensor = torch.zeros_like(toks).squeeze()
            dst_tensor[:len(dst)] = torch.from_numpy(np.array(dst)).to(device=device)
            predictions = model({'x': toks, 'src_len': length, 'dst': dst_tensor.unsqueeze(0)})
            token_offset = len(dst) - 1
            if not sample:
                output = torch.argmax(predictions, -1).squeeze(0)
                output = output[token_offset].item()
            else:

                # using a multinomial distribution to predict the word returned by the model
                predictions = predictions.exp().squeeze(0)
                output = torch.multinomial(predictions, num_samples=1).squeeze(0)[token_offset].item()


            dst.append(output)

            if output == Offsets.EOS or output == EOU or output == Offsets.PAD:
                break
            response.append(index2word.get(dst[-1], '<ERROR>'))
    return response


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
    model.eval()
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
    parser.add_argument("--activation", type=str, default='relu')
    parser.add_argument('--rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8]*8, nargs='+')
    parser.add_argument("--use_cls", type=str2bool, default=False, help="Prepend a [CLS] token on the encoder?")
    parser.add_argument("--go_token", default="<GO>")
    parser.add_argument("--end_token", default="<EOS>")
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--show_query", type=str2bool, default=False, help="Show the original query as well")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
        args.device = torch.device("cuda", 0)

    if os.path.isdir(args.checkpoint):
        checkpoint, _ = find_latest_checkpoint(args.checkpoint)
        logger.warning("Found latest checkpoint %s", checkpoint)
    else:
        checkpoint = args.checkpoint

    cls = None if not args.use_cls else '[CLS]'
    vectorizer = BPEVectorizer1D(model_file=args.subword_model_file, vocab_file=args.subword_vocab_file,
                                 mxlen=args.nctx, emit_begin_tok=cls)
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

    queries = []
    if os.path.exists(args.input) and os.path.isfile(args.input):
        with open(args.input) as rf:
            for line in rf:
                queries.append(line.strip())
    else:
        queries.append(args.input)

    for query in queries:
        output = ' '.join(decode_sentence(model, vectorizer, query.split(), vocab, index2word, args.device,
                                          max_response_length=args.nctx, sou_token=args.go_token,
                                          eou_token=args.end_token,
                                          sample=args.sample))
        output = output.replace('@@ ', '')
        if args.show_query:
            print(f"[Query] {query}")
            print(f"[Response] {output}")
        elif wf:
            wf.write(f'{output}\n')
            wf.flush()
        else:
            print(output)
    if wf:
        wf.close()
run()
