import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import baseline.embeddings
import baseline.pytorch.embeddings
from eight_mile.pytorch.serialize import load_transformer_de_npz
from mead.api_examples.transformer_utils import MultiFileDatasetReader, TransformerBoWPairedModel
from eight_mile.progress import create_progress_bar
from eight_mile.utils import str2bool
from eight_mile.pytorch.layers import find_latest_checkpoint, PairedModel
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger("baseline")


def get_next_k(l, k):
    next_batch_x = []
    next_batch_y = []
    seen_y = set()
    for (xs, ys) in l:
        for x, y in zip(xs, ys):
            s_y = str(y)
            if s_y not in seen_y:
                seen_y.add(s_y)
                next_batch_x.append(x)
                next_batch_y.append(y)
                if len(next_batch_x) == k:
                    seen_y = set()
                    b_x = torch.stack(next_batch_x)
                    b_y = torch.stack(next_batch_y)
                    next_batch_x = []
                    next_batch_y = []
                    yield b_x, b_y

def create_model(model_type, embeddings, d_model, d_ff, num_heads, num_layers, rpr_k, d_k, reduction_d_k,
                 stacking_layers, windowed_ra, reduction_type, logger):

    if model_type == 'transformer-bow':
        model = TransformerBoWPairedModel(embeddings, d_model, d_ff, 0, num_heads, num_layers, rpr_k=rpr_k, d_k=d_k,
                                          reduction_d_k=reduction_d_k, stacking_layers=stacking_layers, ffn_pdrop=0,
                                          reduction_type=reduction_type,
                                          windowed_ra=windowed_ra)


    else:
        model = PairedModel(embeddings, d_model, d_ff, 0, num_heads, num_layers, rpr_k=rpr_k, d_k=d_k,
                            reduction_d_k=reduction_d_k, stacking_layers=stacking_layers,
                            reduction_type=reduction_type,
                            windowed_ra=windowed_ra)

    logger.info(model)
    return model

def main():
    parser = argparse.ArgumentParser("Load a dual-encoder model and do response selection on testing data")
    parser.add_argument("--embed_type", type=str, default='default',
                        choices=["default", "positional", "learned-positional"],
                        help="register label of the embeddings")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--windowed_ra", type=str2bool, default=False, help="whether prevent attention beyond rpr_k")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--file_type", default='json', help="Suffix for data")
    parser.add_argument("--record_keys", default=['x', 'y'], nargs='+')
    parser.add_argument("--model_type", default="dual-encoder", choices=["dual-encoder", "transformer-bow"])
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--subword_model_file", type=str, help="The BPE model file", required=True)
    parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
    parser.add_argument("--reduction_d_k", type=int, default=64, help="Dimensions of Key and Query in the single headed"
                                                                      "reduction layers")
    parser.add_argument("--reduction_type", type=str, default="2ha", help="Method of reduction, defaults to 2-headed attention")
    parser.add_argument("--stacking_layers", type=int, nargs='+',
                        help="Hidden sizes of the dense stack (ff2 from the convert paper)")

    parser.add_argument("--reader_type", type=str, default='preprocessed', choices=['ntp', 'nsp', 'preprocessed', 'tfrecord'])
    parser.add_argument("--output_file", type=str)
    parser.add_argument('--rpr_k',
                        help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--num_test_workers", type=int, default=1, help="Number valid workers")
    parser.add_argument("--ckpt", type=str, help="path to the model checkpoint", required=True)
    parser.add_argument("--test_file", type=str, help="path to the testing data")
    parser.add_argument("--recall_k", type=int, default=100, help="select the response from how many candidates")
    parser.add_argument("--recall_top", type=int, default=1, help="whether the correct response is ranked top x")
    parser.add_argument("--num_batches", type=int, default=1_000_000)
    parser.add_argument("--extra_tokens", help="What extra tokens should we use", nargs="+", default=["[CLS]", "[MASK]"])
    args = parser.parse_args()

    reader = MultiFileDatasetReader(args.nctx, args.nctx, model_file=args.subword_model_file,
                                    vocab_file=args.subword_vocab_file, file_type=args.file_type,
                                    reader_type=args.reader_type, record_keys=args.record_keys,
                                    extra_tokens=args.extra_tokens)

    vocab = reader.build_vocab()
    # If we are not using chars, then use 'x' for both input and output
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, known_vocab=vocab['x'],
                                                       preserve_vocab_indices=True,
                                                       embed_type=args.embed_type)

    vocabs = preproc_data['vocab']
    embeddings = preproc_data['embeddings']
    logger.info("Loaded embeddings")

    test_set = reader.load(args.test_file, vocabs)
    ind2tok = {ind: tok for tok, ind in vocabs.items()}

    # use other samples in a batch as negative samples. Don't shuffle to compare with conveRT benchmarks
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_test_workers)
    logger.info("Loaded datasets")
    model = create_model(args.model_type,
                         embeddings, d_model=args.d_model, d_ff=args.d_ff,
                         num_heads=args.num_heads, num_layers=args.num_layers,
                         rpr_k=args.rpr_k, d_k=args.d_k, reduction_d_k=args.reduction_d_k,
                         stacking_layers=args.stacking_layers, windowed_ra=args.windowed_ra,
                         reduction_type=args.reduction_type,
                         logger=logger)

    if os.path.isdir(args.ckpt):
        checkpoint, _ = find_latest_checkpoint(args.ckpt)
        logger.warning("Found latest checkpoint %s", checkpoint)
    else:
        checkpoint = args.ckpt
    if checkpoint.endswith(".npz"):
        load_transformer_de_npz(model, checkpoint)
    else:
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model.to(args.device)

    numerator = 0
    denominator = 0
    model.eval()
    num_batches = min(len(test_loader), args.num_batches)
    pg = create_progress_bar(num_batches)

    for i, batch in enumerate(get_next_k(test_loader, args.recall_k)):

        if i >= num_batches or batch[0].shape[0] != args.recall_k:
            break
        with torch.no_grad():
            inputs, targets = batch
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            query = model.encode_query(inputs).unsqueeze(1)  # [B, 1, H]
            response = model.encode_response(targets).unsqueeze(0)  # [1, B, H]
            all_score = nn.CosineSimilarity(dim=-1)(query, response)
            _, indices = torch.topk(all_score, args.recall_top, dim=1)
            correct = (indices == torch.arange(args.recall_k, device=all_score.device).unsqueeze(1).expand(-1, args.recall_top)).sum()
            numerator += correct
            print(f"Selected {correct} correct responses out of {args.recall_k}")
            denominator += args.recall_k
        pg.update()
    pg.done()
    acc = float(numerator)/denominator

    print(f"{args.recall_top}@{args.recall_k} acc: {acc}")

    if args.output_file:
        with open(args.output_file, 'a') as wf:
            wf.write(f"Checkpoint: {checkpoint}; {args.recall_top}@{args.recall_k} accuracy: {acc}\n")


if __name__ == '__main__':
    main()
