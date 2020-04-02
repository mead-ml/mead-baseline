import argparse
from transformer_utils import *
from baseline.pytorch.lm import TransformerLanguageModel
from baseline.progress import create_progress_bar
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger("baseline")


def create_model(model_type, embeddings, d_model=512, d_ff=2048, dropout=0., num_heads=8, num_layers=6, rpr_k=[],
                 d_k=None):
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
        model = PairedModel(embeddings, d_model, d_ff, dropout, num_heads, num_layers, rpr_k=rpr_k, d_k=d_k)
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


parser = argparse.ArgumentParser("Load a dual-encoder model and do response selection on testing data")
parser.add_argument("--ckpt", type=str, help="path to the model checkpoint")
parser.add_argument("--test_file", type=str, help="path to the testing data")
parser.add_argument("--model_type", type=str, choices=['dual-encoder', 'encoder-decoder', 'clm'])
parser.add_argument("--subword_model_file", type=str, required=True)
parser.add_argument("--subword_vocab_file", type=str, required=True)
parser.add_argument("--nctx", type=int, default=64, help="Max context length (for both encoder and decoder)")
parser.add_argument("--reader_type", type=str, default='ntp', choices=['ntp', 'nsp'])
parser.add_argument("--embed_type", type=str, default='positional',
                    help="register label of the embeddings, so far support positional or learned-positional")
parser.add_argument("--recall_k", type=int, default=100, help="select the response from how many candidates")
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

# use other samples in a batch as negative samples, reference usually reports selection acc from 100 samples, so hard
# coded here. Don't shuffle to compare with conveRT benchmarks
test_loader = DataLoader(test_set, batch_size=args.recall_k, num_workers=4)
logger.info("Loaded datasets")

model = create_model(args.model_type, embeddings)
model.load_state_dict(torch.load(args.ckpt))
model.to(args.device)
loss_function = model.create_loss()
loss_function.to(args.device)

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
            all_score = nn.CosineSimilarity(dim=-1)(query, response)

        elif args.model_type == 'clm':
            contexts, responses = batch
            # expand the tensors so each response is paired to each context
            # [B*B, T] make B copies of each sample
            contexts = contexts.expand(args.recall_k, -1, -1).transpose(0, 1).reshape(-1, args.nctx)
            # [B*B, T] make B copies of the whole batch
            responses = responses.expand(args.recall_k, -1, -1).reshape(-1, args.nctx)
            context_lengths = torch.sum(contexts != 0, 1)
            expand_batchsz = args.recall_k**2
            losses = torch.zeros(expand_batchsz).to(args.device)
            for i in range(expand_batchsz):
                sequence = torch.cat((contexts[i, :context_lengths[i]], responses[i], contexts[i, context_lengths[i]:]))
                sequence = sequence.unsqueeze(0)  # make a batch with bsz=1
                inputs = {'x': sequence.to(args.device)}
                labels = sequence.to(args.device)
                labels = labels.transpose(0, 1).contiguous()
                logits = model(inputs, None)[0].transpose(0, 1).contiguous()
                shift_logits = logits[:-1]
                shift_labels = labels[1:]
                losses[i] = loss_function(shift_logits, shift_labels)
            all_score = - losses.reshape(args.recall_k, -1)
            print(all_score)

        elif args.model_type == 'encoder-decoder':
            pass

        _, max_indices = torch.max(all_score, 1)
        numerator += (max_indices == torch.arange(args.recall_k).to(args.device)).sum()
        denominator += args.recall_k
    pg.update()
pg.done()
acc = float(numerator)/denominator

print(f"The 1@100 accuracy is {acc}")

