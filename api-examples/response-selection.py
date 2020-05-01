import argparse
import os
from transformer_utils import *
from baseline.pytorch.lm import TransformerLanguageModel
from baseline.progress import create_progress_bar
from baseline.utils import str2bool
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger("baseline")


parser = argparse.ArgumentParser("Load a dual-encoder model (conveRT) and do response selection on testing data")
parser.add_argument("--d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
parser.add_argument("--d_ff", type=int, default=2048, help="FFN dimension")
parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
parser.add_argument("--num_ft_workers", type=int, default=4, help="Number train workers")
parser.add_argument("--num_test_workers", type=int, default=2, help="Number valid workers")
parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
parser.add_argument("--nctx", type=int, default=64, help="Max context length (for both encoder and decoder)")
parser.add_argument("--embed_type", type=str, default='positional',
                    help="register label of the embeddings, so far support positional or learned-positional")
parser.add_argument("--stacking_layers", type=int, nargs='+', default=[1024, 1024, 1024])
parser.add_argument('--rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                    type=int, default=[3, 5, 48, 48, 48, 48], nargs='+')
parser.add_argument("--reduction_d_k", type=int, default=64, help="Dimensions of Key and Query in the single headed"
                                                                  "reduction layers")
parser.add_argument("--ckpt", type=str, help="path to the model checkpoint")
parser.add_argument("--test_file", type=str, help="path to the testing data")
parser.add_argument("--subword_model_file", type=str, required=True)
parser.add_argument("--subword_vocab_file", type=str, required=True)
parser.add_argument("--recall_k", type=int, default=100, help="select the response from how many candidates")
parser.add_argument("--recall_top", type=int, default=1, help="whether the correct response is ranked top x")
parser.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device (cuda or cpu)")
args = parser.parse_args()

reader = MultiFileDatasetReader(args.nctx, args.subword_model_file, args.subword_vocab_file, '*.txt', 'ntp')
vocab = reader.build_vocab()

preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, known_vocab=vocab['x'], embed_type=args.embed_type)
vocabs = preproc_data['vocab']
embeddings = preproc_data['embeddings']
logger.info("Loaded embeddings")

test_set = reader.load(args.test_file, vocabs)
ind2tok = {ind: tok for tok, ind in vocabs.items()}

# use other samples in a batch as negative samples. Don't shuffle to compare with conveRT benchmarks
test_loader = DataLoader(test_set, batch_size=args.recall_k, num_workers=args.num_test_workers)
logger.info("Loaded datasets")

model = create_model(embeddings,
                     model_type='dual-encoder',
                     d_model=args.d_model,
                     d_ff=args.d_ff,
                     num_heads=args.num_heads,
                     num_layers=args.num_layers,
                     stacking_layers=args.stacking_layers,
                     dropout=0.,
                     rpr_k=args.rpr_k,
                     d_k=args.d_k,
                     reduction_d_k=args.reduction_d_k,
                     ff_pdrop=0.,
                     logger=logger)

if os.path.isdir(args.ckpt):
    checkpoint = find_latest_checkpoint(args.ckpt)
    logger.warning("Found latest checkpoint %s", checkpoint)
else:
    checkpoint = args.ckpt
model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
model.to(args.device)

numerator = 0
denominator = 0
model.eval()
pg = create_progress_bar(len(test_loader)//args.recall_k)
for batch in test_loader:
    if batch[0].shape[0] != args.recall_k:
        break
    with torch.no_grad():
        x, y = batch
        inputs = x.to(args.device)
        targets = y.to(args.device)
        query = model.encode_query(inputs).unsqueeze(1)  # [B, 1, H]
        response = model.encode_response(targets).unsqueeze(0)  # [1, B, H]
        all_score = nn.CosineSimilarity(dim=-1)(query, response).to('cpu')

        _, indices = torch.topk(all_score, args.recall_top, dim=1)
        correct = (indices == torch.arange(args.recall_k).unsqueeze(1).expand(-1, args.recall_top)).sum()
        numerator += correct
        print(f"Selected {correct} correct responses out of {args.recall_k}")
        denominator += args.recall_k
    pg.update()
pg.done()
acc = float(numerator)/denominator

print(f"{args.recall_top}@{args.recall_k} acc: {acc}")
with open('./results.txt', 'a') as wf:
    wf.write(f"Checkpoint: {checkpoint}; {args.recall_top}@{args.recall_k} accuracy: {acc}\n")
