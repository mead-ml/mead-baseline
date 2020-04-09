import argparse
from transformer_utils import *
from baseline.pytorch.lm import TransformerLanguageModel
from baseline.progress import create_progress_bar
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger("baseline")


def create_model(model_type, embeddings, d_model=512, d_ff=2048, dropout=0., num_heads=8, num_layers=6,
                 stacking_layers=None, rpr_k=[], d_k=None):
    if len(rpr_k) == 0 or rpr_k[0] < 1:
        rpr_k = None
    if model_type == "encoder-decoder":
        logger.info("Creating tied encoder decoder model")
        hps = {"dsz": d_model,
               "hsz": d_model,
               "d_ff": d_ff,
               "dropout": dropout,
               "num_heads": num_heads,
               "layers": num_layers,
               "encoder_type": "transformer",
               "decoder_type": "transformer",
               "src_lengths_key": "x_lengths",
               "d_k": d_k,
               "rpr_k": rpr_k}
        model = TiedEmbeddingsSeq2SeqModel(embeddings, **hps)
    elif model_type == 'dual-encoder':
        model = PairedModel(embeddings, d_model, d_ff, dropout, num_heads, num_layers, stacking_layers=stacking_layers,
                            rpr_k=rpr_k, d_k=d_k)
    else:
        model = TransformerLanguageModel.create({'x': embeddings},
                                                hsz=d_model,
                                                d_ff=d_ff,
                                                tie_weights=True,
                                                dropout=dropout,
                                                gpu=False,
                                                num_heads=num_heads,
                                                layers=num_layers,
                                                rpr_k=rpr_k,
                                                d_k=d_k,
                                                src_keys=['x'],
                                                tgt_key='x')
    # logger.info(model)
    return model


def get_joint_logprob(model: nn.Module, sequences: torch.Tensor, context_length: int):
    mask = (sequences != 0).to(args.device)
    inputs = {'x': sequences.to(args.device)}
    targets = sequences.to(args.device)
    softmaxes = model.predict(inputs, numpy_to_tensor=False)
    softmaxes = softmaxes[:, :-1]
    targets = targets[:, 1:]
    token_logprobs = torch.log(torch.gather(softmaxes, -1, targets.unsqueeze(-1)).squeeze(-1))
    # for j in range(args.recall_k):
    #     print(sequences[j, :32])
    #     tokens = [ind2tok[i] for i in sequences[j, :32].numpy()]
    #     probs = [p for p in token_probs[j, :32].numpy()]
    #     print(list(zip(tokens, probs)))
    token_logprobs = token_logprobs*mask[:, 1:]
    return token_logprobs[:, context_length:].sum(axis=1)


parser = argparse.ArgumentParser("Load a dual-encoder model and do response selection on testing data")
parser.add_argument("--model_type", type=str, choices=['dual-encoder', 'encoder-decoder', 'clm'])
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
parser.add_argument("--stacking_layers", type=int, nargs='+', default=[512, 512, 512])
parser.add_argument('--rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                    type=int, default=[3, 5, 48, 48, 48, 48], nargs='+')
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

preproc_data = baseline.embeddings.load_embeddings('x', dsz=512, known_vocab=vocab['x'], embed_type=args.embed_type)
vocabs = preproc_data['vocab']
embeddings = preproc_data['embeddings']
logger.info("Loaded embeddings")

test_set = reader.load(args.test_file, vocabs)
ind2tok = {ind: tok for tok, ind in vocabs.items()}

# use other samples in a batch as negative samples. Don't shuffle to compare with conveRT benchmarks
test_loader = DataLoader(test_set, batch_size=args.recall_k, num_workers=args.num_test_workers)
logger.info("Loaded datasets")

model = create_model(args.model_type,
                     embeddings,
                     d_model=args.d_model,
                     d_ff=args.d_ff,
                     dropout=0.,
                     num_heads=args.num_heads,
                     num_layers=args.num_layers,
                     stacking_layers=args.stacking_layers,
                     rpr_k=args.rpr_k,
                     d_k=args.d_k)
model.load_state_dict(torch.load(args.ckpt))
model.to(args.device)

numerator = 0
denominator = 0
model.eval()
pg = create_progress_bar(len(test_loader)//args.recall_k)
for batch in test_loader:
    if batch[0].shape[0] != args.recall_k:
        break
    with torch.no_grad():
        if args.model_type == 'dual-encoder':
            x, y = batch
            inputs = x.to(args.device)
            targets = y.to(args.device)
            query = model.encode_query(inputs).unsqueeze(1)  # [B, 1, H]
            response = model.encode_response(targets).unsqueeze(0)  # [1, B, H]
            all_score = nn.CosineSimilarity(dim=-1)(query, response).to('cpu')

        elif args.model_type == 'clm':
            contexts, responses = batch
            context_lengths = torch.sum(contexts != 0, 1)
            response_lengths = torch.sum(responses != 0, 1).to(args.device)
            scores = []
            for i in range(args.recall_k):
                # make recall_k copies of one context and remove padding
                context = contexts[i][:context_lengths[i]].expand(args.recall_k, -1)
                # pair the same context with all candidate responses
                sequences = torch.cat([context, responses], axis=1)
                total_logprobs = get_joint_logprob(model, sequences, context_lengths[i])
                response_logprobs = get_joint_logprob(model, responses, 0)
                scores.append((total_logprobs - response_logprobs).to('cpu'))
            all_score = torch.stack(scores, axis=0)

        elif args.model_type == 'encoder-decoder':
            pass

        # _, max_indices = torch.max(all_score, 1)
        _, indices = torch.topk(all_score, args.recall_top, dim=1)
        correct = (indices == torch.arange(args.recall_k).unsqueeze(1).expand(-1, args.recall_top)).sum()
        numerator += correct
        print(f"Selected {correct} correct responses out of {args.recall_k}")
        denominator += args.recall_k
    pg.update()
pg.done()
acc = float(numerator)/denominator

print(f"The 1@{args.recall_k} accuracy is {acc}")

