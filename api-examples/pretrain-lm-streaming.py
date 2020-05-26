import logging
import time
import os
from argparse import ArgumentParser
import tempfile
import baseline
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from eight_mile.utils import str2bool, write_json, Offsets
import baseline.pytorch.embeddings
import baseline.embeddings
from eight_mile.optz import *
from eight_mile.pytorch.layers import checkpoint_for, rm_old_checkpoints, Average
from eight_mile.pytorch.optz import *
from eight_mile.pytorch.serialize import save_tlm_npz
from baseline.pytorch.lm import TransformerLanguageModel
from transformer_utils import MultiFileDatasetReader, on_demand_mlm_masking

logger = logging.getLogger(__file__)


def save_checkpoint(model: torch.nn.Module, model_base: str, count: int, tick_type: str = 'epoch'):

    checkpoint_name = checkpoint_for(model_base, count, tick_type=tick_type)
    # Its possible due to how its called that we might save the same checkpoint twice if we dont check first
    if os.path.exists(checkpoint_name):
        logger.info("Checkpoint already exists: %s", checkpoint_name)
        return
    logger.info("Creating checkpoint: %s", checkpoint_name)
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), checkpoint_name)
    else:
        torch.save(model.state_dict(), checkpoint_name)

    rm_old_checkpoints(model_base, count)


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_file", type=str, required=True, help='File path to use for train file')
    parser.add_argument("--valid_file", type=str, required=True, help='File path to use for valid file')
    parser.add_argument("--dataset_cache", type=str, default=os.path.expanduser('~/.bl-data'),
                        help="Path or url of the dataset cache")
    parser.add_argument("--dataset_key", default="reddit",
                        help="dataset key for basedir")
    parser.add_argument("--embed_type", type=str, default='positional',
                        help="register label of the embeddings, so far support positional or learned-positional")
    parser.add_argument("--d_model", type=int, default=410, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2100, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--num_valid_workers", type=int, default=2, help="Number valid workers")
    parser.add_argument("--nctx", type=int, default=128, help="Max input length")
    parser.add_argument("--pattern", default='*.txt', help="Glob pattern for data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    parser.add_argument("--subword_model_file", type=str, help="The BPE model file", required=True)
    parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--optim", default="adam", type=str, help="Optimizer to use (defaults to adam)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--restart_tt", type=str, help="Optional param for legacy checkpoints (step|epoch)")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--update_steps", type=int, default=100, help="The number of steps to take before saving a checkpoint")
    parser.add_argument("--mlm", type=str2bool, default=True, help="Use Masked Language Model (MLM) objective")
    parser.add_argument("--preprocessed", type=str2bool, default=False, help="Has the data already been preprocessed?")

    parser.add_argument('--rpr_k',
                        help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[3, 5, 48, 48, 48, 48], nargs='+')
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--distributed",
                        type=str2bool,
                        default=False,
                        help="Are we doing distributed training?")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Local rank for distributed training (-1 means use the environment variables to find)")
    parser.add_argument("--chars_per_word",
                        type=int,
                        default=40,
                        help="How many max characters per word")

    args = parser.parse_args()

    if args.basedir is None:
        args.basedir = 'lm-{}-bpe-{}'.format(args.dataset_key, os.getpid())
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("Cache directory [%s]", args.dataset_cache)

    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

    do_on_demand_masking = args.mlm and not args.preprocessed

    if args.distributed:
        if args.local_rank == -1:
            # https://github.com/kubeflow/pytorch-operator/issues/128
            # https://github.com/pytorch/examples/blob/master/imagenet/main.py
            logger.info("Setting local rank to RANK env variable")
            args.local_rank = int(os.environ['RANK'])
        logger.warning("Local rank (%d)", args.local_rank)
        # In an env like k8s with kubeflow each worker will only see a single gpu
        # with an id of 0. If the gpu count is 1 then we are probably in an env like
        # that so we should just use the first (and only) gpu avaiable
        if torch.cuda.device_count() == 1:
            torch.cuda.set_device(0)
            args.device = torch.device("cuda", 0)
        # This program assumes multiprocess/multi-device on a single node. Each
        # process gets a rank (via cli or ENV variable) and uses that rank to select
        # which gpu to use. This only makes sense on a single node, if you had 4
        # processes on 2 nodes where each node has 2 GPUs then the ranks would be
        # 0, 1, 2, 3 but the gpus numbers would be node 0: 0, 1 and node 1: 0, 1
        # and this assignment to gpu 3 would fail. On a single node with 4 processes
        # and 4 gpus the rank and gpu ids will align and this will work
        else:
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    reader_type = "lang" if not args.preprocessed else "preprocessed"
    reader = MultiFileDatasetReader(args.nctx, args.subword_model_file, args.subword_vocab_file, args.pattern,
                                    reader_type=reader_type)

    # This looks a bit funny but the streaming reader ignores our vocab and gives us the one from the subword_model
    # However, we do need to get counts from our dataset for validation so we can calculate the perplexity
    vocab = reader.build_vocab([args.valid_file])
    # If we are not using chars, then use 'x' for both input and output
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, known_vocab=vocab['x'],
                                                       embed_type=args.embed_type)
    vocabs = preproc_data['vocab']

    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(args.basedir, 'vocabs.json'))
    embeddings = {'x': preproc_data['embeddings']}
    logger.info("Loaded embeddings")

    train_set = reader.load(args.train_file, vocabs)
    valid_set = reader.load(args.valid_file, vocabs)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_train_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_valid_workers)
    logger.info("Loaded datasets")
    logger.info("Using embedding type [%s]", args.embed_type)

    if args.mlm:
        mask_from = vocabs
        vocab_size = len(mask_from)
        mask_value = mask_from.get("[MASK]", mask_from.get("<MASK>", -1))
        if mask_value == -1:
            logger.error("We could not find a suitable masking token in the vocab")
            return

    if len(args.rpr_k) == 0 or args.rpr_k[0] < 1:
        rpr_k = None
    else:
        rpr_k = args.rpr_k

    if args.mlm:
        from baseline.pytorch.lm import TransformerMaskedLanguageModel
        model = TransformerMaskedLanguageModel.create(embeddings,
                                                      hsz=args.d_model,
                                                      d_ff=args.d_ff,
                                                      tie_weights=True,
                                                      dropout=args.dropout,
                                                      gpu=False,
                                                      num_heads=args.num_heads,
                                                      layers=args.num_layers,
                                                      rpr_k=rpr_k,
                                                      d_k=args.d_k,
                                                      src_keys=['x'], tgt_key='x')
    else:
        model = TransformerLanguageModel.create(embeddings,
                                                hsz=args.d_model,
                                                d_ff=args.d_ff,
                                                tie_weights=True,
                                                dropout=args.dropout,
                                                gpu=False,
                                                num_heads=args.num_heads,
                                                layers=args.num_layers,
                                                rpr_k=rpr_k,
                                                d_k=args.d_k,
                                                src_keys=['x'], tgt_key='x')
    model.to(args.device)
    loss_function = model.create_loss()
    loss_function.to(args.device)

    logger.info("Loaded model and loss")

    # according to pytorch, len(train_loader) will return len(train_set) when train_set is IterableDataset, so manually
    # correct it here
    steps_per_epoch = len(train_loader) // (args.batch_size*num_gpus)
    update_on = steps_per_epoch // args.update_steps
    report_on = update_on // 10
    logger.info(f"Steps per epoch per GPU: {steps_per_epoch}. Saving checkpoint every {update_on} steps.")
    cosine_decay = CosineDecaySchedulerPyTorch(steps_per_epoch * args.epochs, lr=args.lr)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, cosine_decay, lr=args.lr)

    global_step = 0
    start_epoch = 0
    if args.restart_from:
        model.load_state_dict(torch.load(args.restart_from))
        vec = args.restart_from.split("-")

        if args.restart_tt:
            tick_type = args.restart_tt
        else:
            tick_type = vec[-2]
        step_num = int(vec[-1].split(".")[0])
        if tick_type == 'epoch':
            start_epoch = step_num
            global_step = start_epoch * steps_per_epoch

        else:
            start_epoch = step_num // steps_per_epoch
            global_step = step_num

        logger.info("Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d, epoch=%d",
                    args.restart_from, global_step, start_epoch+1)

    optimizer = OptimizerManager(model, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
    logger.info("Model has {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Prepare model for distributed training if needed
    if args.distributed:
        # This program assume pure data parallelism, each model is on a single gpu
        # If we wanted to support model and data parallelism we would need to update
        # the selection of gpus based on rank, it would need to select multiple ids
        # based on rank, here we select only a single gpu and use it for input and
        # output.
        model = DistributedDataParallel(model, device_ids=[args.device], output_device=args.device)
        logger.info("Model located on %s", args.device)

    # This is the training loop
    steps = global_step
    model_base = os.path.join(args.basedir, 'checkpoint')

    for epoch in range(start_epoch, args.epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}
        optimizer.zero_grad()
        start = time.time()
        model.train()
        for i, batch in enumerate(train_loader):
            steps += 1
            x, y = batch
            inputs = x.to(args.device)
            labels = y.to(args.device)
            if do_on_demand_masking:
                inputs, labels = on_demand_mlm_masking(inputs, labels, mask_value, vocab_size)
            inputs = {'x': inputs}

            labels = labels.transpose(0, 1).contiguous().to(args.device)
            logits = model(inputs, None)[0].transpose(0, 1).contiguous()
            if args.mlm:
                loss = loss_function(logits, labels)
            else:
                shift_logits = logits[:-1]
                shift_labels = labels[1:]
                loss = loss_function(shift_logits, shift_labels)
            loss.backward()
            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % report_on == 0:
                logging.info(avg_loss)
            if (i + 1) % update_on == 0 and args.local_rank < 1:
                elapsed = (time.time() - start)/60
                logging.info('elapsed time this epoch %d min', elapsed)
                logging.info('elapsed step time %f steps/min', i/elapsed)
                save_checkpoint(model, model_base, steps, tick_type='step')

        # How much time elapsed in minutes
        elapsed = (time.time() - start)/60
        train_token_loss = avg_loss.avg
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        train_token_ppl = math.exp(train_token_loss)
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_token_loss
        metrics['train_ppl'] = train_token_ppl
        avg_valid_loss = Average('average_valid_loss')
        start = time.time()
        model.eval()
        for batch in valid_loader:
            with torch.no_grad():
                x, y = batch
                inputs = x.to(args.device)
                labels = y.to(args.device)

                if do_on_demand_masking:
                    inputs, labels = on_demand_mlm_masking(inputs, labels, mask_value, vocab_size)
                inputs = {'x': inputs}
                labels = labels.transpose(0, 1).contiguous()
                logits = model(inputs, None)[0].transpose(0, 1).contiguous()
                if args.mlm:
                    loss = loss_function(logits, labels)
                else:
                    shift_logits = logits[:-1]
                    shift_labels = labels[1:]
                    loss = loss_function(shift_logits, shift_labels)
                avg_valid_loss.update(loss.item())

        valid_token_loss = avg_valid_loss.avg
        valid_token_ppl = math.exp(valid_token_loss)

        elapsed = (time.time() - start)/60
        metrics['valid_elapsed_min'] = elapsed

        metrics['average_valid_loss'] = valid_token_loss
        metrics['average_valid_word_ppl'] = valid_token_ppl
        logger.info(metrics)

        if args.local_rank < 1:

            # Should probably do this more often
            checkpoint_name = checkpoint_for(model_base, epoch)
            logger.info("Creating checkpoint: %s", checkpoint_name)
            if args.distributed:
                torch.save(model.module.state_dict(), checkpoint_name+'.pth')
                save_tlm_npz(model.module, checkpoint_name+'.npz')
            else:
                torch.save(model.state_dict(), checkpoint_name+'.pth')
                save_tlm_npz(model, checkpoint_name+'.npz')

            rm_old_checkpoints(model_base, epoch)


if __name__ == "__main__":
    train()

