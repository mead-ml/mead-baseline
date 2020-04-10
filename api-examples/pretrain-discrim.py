import logging
import time
import os
from argparse import ArgumentParser
import baseline
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, write_json
import glob
from baseline.pytorch.embeddings import *
import baseline.embeddings
from eight_mile.optz import *
from eight_mile.pytorch.optz import *
from eight_mile.pytorch.layers import Average, checkpoint_for, rm_old_checkpoints
logger = logging.getLogger(__file__)
from baseline.pytorch.lm import TransformerMaskedLanguageModel
from eight_mile.pytorch.layers import TransformerEncoderStack, EmbeddingsStack, subsequent_mask
from transformer_utils import MultiFileDatasetReader, TransformerDiscriminator

"""Pre-train an discriminator Transformer model in PyTorch

This file uses Baseline to train a Transformer-based discriminative model
model, similar to (https://openreview.net/pdf?id=r1xMH1BtvB)
"""


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

def create_generator(embeddings, d_model, d_ff, dropout, num_heads, num_layers, rpr_k, d_k):
    """Produces an MLM generator model

    :param embeddings: The pre-inited embeddings
    :param d_model: The size of the model
    :param d_ff: The feed-forward layer size
    :param dropout: The amount of dropout
    :param num_heads: The number of attention heads
    :param num_layers: The number of layers
    :param rpr_k: The relative position sizes
    :param d_k: The head size if single headed
    :return:
    """
    model = TransformerMaskedLanguageModel.create(embeddings,
                                                  hsz=d_model,
                                                  d_ff=d_ff,
                                                  tie_weights=True,
                                                  dropout=dropout,
                                                  gpu=False,
                                                  num_heads=num_heads,
                                                  layers=num_layers,
                                                  rpr_k=rpr_k,
                                                  d_k=d_k,
                                                  src_keys=['x'], tgt_key='x')
    return model


def best_from(x_preds):
    return x_preds.argmax(axis=-1)

def get_accuracy(preds, true_or_fake, logits):
    flat_logits = logits.reshape(-1)
    nz_preds = preds.view(-1)[flat_logits != 0]
    nz_true_or_fake = true_or_fake.view(-1)[flat_logits != 0]

    preds_true = (nz_preds > 0.5).squeeze().to(nz_true_or_fake.dtype)
    num = torch.sum((nz_true_or_fake == preds_true).to(torch.float32))
    denom = nz_true_or_fake.nelement()
    return num / denom


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_file", type=str, help='Optional file path to use for train file')
    parser.add_argument("--valid_file", type=str, help='Optional file path to use for valid file')
    parser.add_argument("--gen_d_model", type=int, default=256, help="Model dimension (and embedding dsz)")
    parser.add_argument("--discrim_d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--gen_d_ff", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--discrim_d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--gen_d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--discrim_d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--gen_num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--discrim_num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--gen_num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--discrim_num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--gen_dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--discrim_dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument('--gen_rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[3, 5, 48, 48, 48, 48], nargs='+')

    parser.add_argument('--discrim_rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[3, 5, 48, 48, 48, 48], nargs='+')

    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--num_valid_workers", type=int, default=2, help="Number valid workers")
    parser.add_argument("--nctx", type=int, default=64, help="Max context length (for both encoder and decoder)")
    parser.add_argument("--embed_type", type=str, default='positional',
                        help="register label of the embeddings, so far support positional or learned-positional")
    parser.add_argument("--pattern", default='*.txt', help="Glob pattern for data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--dataset_key", default="reddit",
                        help="dataset key for basedir")
    parser.add_argument("--subword_model_file", type=str, required=True)
    parser.add_argument("--subword_vocab_file", type=str, required=True)
    parser.add_argument("--optim", default="adam", type=str, help="Optimizer to use (defaults to adam)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--restart_tt", type=str, help="Optional param for legacy checkpoints (step|epoch)")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--update_steps", type=int, default=100, help="The number of steps to take before saving a checkpoint")
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

    args = parser.parse_args()

    if args.train_file and not args.valid_file:
        logger.error("If you provide a train_file, you must provide a valid_file")
        return

    if not args.train_file and args.valid_file:
        logger.error("If you provide a valid_file, you must also provide a train_file")
        return

    if args.basedir is None:
        args.basedir = 'gd-{}-bpe-{}'.format(args.dataset_key, os.getpid())
    logging.basicConfig(
        format="%(name)s: %(levelname)s: %(message)s",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

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


    reader = MultiFileDatasetReader(args.nctx, args.subword_model_file, args.subword_vocab_file, args.pattern,
                                    reader_type="lang")

    # This looks a bit funny but the streaming reader ignores our vocab and gives us the one from the subword_model
    # However, we do need to get counts from our dataset for validation so we can calculate the perplexity
    vocab = reader.build_vocab([args.valid_file])
    # If we are not using chars, then use 'x' for both input and output
    gen_embed = baseline.embeddings.load_embeddings('x', dsz=args.gen_d_model, known_vocab=vocab['x'],
                                                    embed_type=args.embed_type)
    vocabs = gen_embed['vocab']
    discrim_embed = baseline.embeddings.load_embeddings('x', dsz=args.discrim_d_model, known_vocab=vocab['x'],
                                                        embed_type=args.embed_type)

    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(args.basedir, 'vocabs.json'))
    gen_embeddings = {'x': gen_embed['embeddings']}
    discrim_embeddings = {'x': discrim_embed['embeddings']}
    logger.info("Loaded embeddings")

    train_set = reader.load(args.train_file, vocabs)
    valid_set = reader.load(args.valid_file, vocabs)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0)#args.num_train_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=0)#args.num_valid_workers)
    logger.info("Loaded datasets")
    logger.info("Using embedding type [%s]", args.embed_type)

    mask_value = vocabs.get("[MASK]", vocabs.get("<MASK>", -1))
    if mask_value == -1:
        logger.error("We could not find a suitable masking token in the vocab")
        return
    os.makedirs(args.basedir, exist_ok=True)
    vocab_size = len(vocabs)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs['x'], os.path.join(args.basedir, 'vocabs.json'))
    # Get this from a YAML file?
    logger.info("Loaded embeddings")
    logger.info("Loaded datasets")

    if len(args.gen_rpr_k) == 0 or args.gen_rpr_k[0] < 1:
        gen_rpr_k = None
    else:
        gen_rpr_k = args.gen_rpr_k

    if len(args.gen_rpr_k) == 0 or args.discrim_rpr_k[0] < 1:
        discrim_rpr_k = None
    else:
        discrim_rpr_k = args.discrim_rpr_k

    gen_model = create_generator(gen_embeddings, args.gen_d_model, args.gen_d_ff, args.gen_dropout, args.gen_num_heads,
                                 args.gen_num_layers, gen_rpr_k, args.gen_d_k)
    discrim_model = TransformerDiscriminator(discrim_embeddings, args.discrim_d_model, args.discrim_d_ff,
                                             args.discrim_dropout, args.discrim_num_heads, args.discrim_num_layers,
                                             discrim_rpr_k, args.discrim_d_k)
    gen_model.to(args.device)
    gen_loss_fn = gen_model.create_loss()

    discrim_model.to(args.device)
    discrim_loss_fn = discrim_model.create_loss()
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
    #if args.restart_from:
    #    gen_model.load_state_dict(torch.load(args.restart_from))
    #    discrim_model.load_state_dict(torch.load(args.restart_from))
    #    vec = args.restart_from.split("-")
    #    if args.restart_tt:
    #        tick_type = args.restart_tt
    #    else:
    #        tick_type = vec[-2]
    #    step_num = int(vec[-1].split(".")[0])
    #    if tick_type == 'epoch':
    #        start_epoch = step_num
    #        global_step = start_epoch * steps_per_epoch
    #    else:
    #        start_epoch = step_num // steps_per_epoch
    #        global_step = step_num
    #
    #    logger.info("Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d, epoch=%d",
    #                args.restart_from, global_step, start_epoch+1)

    discrim_optz = OptimizerManager(discrim_model, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
    gen_optz = OptimizerManager(gen_model, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
    logger.info("Generator has {:,} parameters".format(sum(p.numel() for p in gen_model.parameters() if p.requires_grad)))
    logger.info("Discriminator has {:,} parameters".format(sum(p.numel() for p in discrim_model.parameters() if p.requires_grad)))

    # Prepare model for distributed training if needed
    if args.distributed:
        # This program assume pure data parallelism, each model is on a single gpu
        # If we wanted to support model and data parallelism we would need to update
        # the selection of gpus based on rank, it would need to select multiple ids
        # based on rank, here we select only a single gpu and use it for input and
        # output.
        gen_model = DistributedDataParallel(gen_model, device_ids=[args.device], output_device=args.device)
        discrim_model = DistributedDataParallel(discrim_model, device_ids=[args.device], output_device=args.device)
        logger.info("Model located on %s", args.device)

    # This is the training loop
    steps = global_step
    model_base = os.path.join(args.basedir, 'checkpoint')
    discrim_base = f'{model_base}-discrim'
    gen_base = f'{model_base}-gen'
    for epoch in range(start_epoch, args.epochs):
        gen_model.train()
        discrim_model.train()
        avg_gen_loss = Average('average_train_gen_loss')
        avg_discrim_loss = Average('average_train_discrim_loss')
        avg_discrim_acc = Average('average_train_discrim_acc')
        metrics = {}
        gen_optz.zero_grad()
        start = time.time()
        for i, batch in enumerate(train_loader):
            steps += 1
            x, y = batch
            noised_x = x.to(args.device)
            labels = y.to(args.device)
            # Replace 15% of tokens
            masked_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).type(torch.bool)
            # Anything not masked is 0 so no loss
            labels[~masked_indices] = 0
            # Of the masked items, mask 80% of them with [MASK]
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool) & masked_indices
            noised_x[indices_replaced] = mask_value
            # Replace 10% of them with random words, rest preserved for auto-encoding
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.bool) & masked_indices & ~indices_replaced
            random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=args.device)
            noised_x[indices_random] = random_words[indices_random]
            labels = labels.transpose(0, 1).contiguous()
            logits = gen_model({'x': noised_x}, None)[0].transpose(0, 1).contiguous()
            gen_loss_step = gen_loss_fn(logits, labels)
            gen_loss_step.backward()
            avg_gen_loss.update(gen_loss_step.item())
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), args.clip)
            gen_optz.step()
            gen_optz.zero_grad()
            labels = y.to(args.device).transpose(0, 1)

            # The logits needs to be replaced with either argmax or sampling.  Which?
            recon_labels = best_from(logits)
            true_or_fake = (recon_labels == labels).to(torch.float32)
            logits = discrim_model({'x': recon_labels})
            discrim_loss_step = discrim_loss_fn(logits.squeeze(), true_or_fake)
            discrim_loss_step.backward()
            avg_discrim_loss.update(discrim_loss_step.item())
            avg_discrim_acc.update(get_accuracy(logits, true_or_fake, labels))
            torch.nn.utils.clip_grad_norm_(discrim_model.parameters(), args.clip)
            discrim_optz.step()
            discrim_optz.zero_grad()
            if (i + 1) % report_on == 0:
                logging.info('Loss g=%f, d=%f, Per token acc=%f', avg_gen_loss.avg, avg_discrim_loss.avg, avg_discrim_acc.avg)
            if (i + 1) % update_on == 0 and args.local_rank < 1:
                elapsed = (time.time() - start)/60
                logging.info('elapsed time this epoch %d min', elapsed)
                logging.info('elapsed step time %f steps/min', i/elapsed)
                save_checkpoint(gen_model, gen_base, steps, tick_type='step')
                save_checkpoint(discrim_model, discrim_base, steps, tick_type='step')
        # How much time elapsed in minutes
        elapsed = (time.time() - start)/60
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_gen_loss'] = avg_gen_loss.avg
        metrics['average_train_discrim_loss'] = avg_discrim_loss.avg
        metrics['average_train_discrim_per_token_accuracy'] = avg_discrim_acc.avg

        avg_valid_gen_loss = Average('average_valid_gen_loss')
        avg_valid_discrim_loss = Average('average_valid_discrim_loss')
        avg_valid_discrim_acc = Average('average_valid_discrim_acc')
        start = time.time()
        gen_model.eval()
        discrim_model.eval()
        for batch in valid_loader:
            with torch.no_grad():
                x, y = batch
                noised_x = x.to(args.device)
                labels = y.to(args.device)
                # Replace 15% of tokens
                masked_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).type(torch.bool)
                # Anything not masked is 0 so no loss
                labels[~masked_indices] = 0
                # Of the masked items, mask 80% of them with [MASK]
                indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.bool) & masked_indices
                noised_x[indices_replaced] = mask_value
                # Replace 10% of them with random work
                indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.bool) & masked_indices & ~indices_replaced
                random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=args.device)
                noised_x[indices_random] = random_words[indices_random]
                labels = labels.transpose(0, 1).contiguous()
                logits = gen_model({'x': noised_x}, None)[0].transpose(0, 1).contiguous()

                gen_loss_step = gen_loss_fn(logits, labels)
                avg_valid_gen_loss.update(gen_loss_step.item())

                labels = y.to(args.device).transpose(0, 1).contiguous()
                recon_labels = best_from(logits)
                logits = discrim_model({'x': recon_labels})
                true_or_fake = (recon_labels == labels).to(torch.float32)
                discrim_loss_step = discrim_loss_fn(logits.squeeze(), true_or_fake)
                avg_valid_discrim_acc.update(get_accuracy(logits, true_or_fake, labels))
                avg_valid_discrim_loss.update(discrim_loss_step.item())

        elapsed = (time.time() - start)/60
        metrics['valid_elapsed_min'] = elapsed
        metrics['average_valid_gen_loss'] = avg_valid_gen_loss.avg
        metrics['average_valid_discrim_loss'] = avg_valid_discrim_loss.avg
        metrics['average_valid_discrim_per_token_accuracy'] = avg_valid_discrim_acc.avg
        logger.info(metrics)

        if args.local_rank < 1:

            # Should probably do this more often
            gen_checkpoint_name = checkpoint_for(gen_base, epoch)
            discrim_checkpoint_name = checkpoint_for(discrim_base, epoch)
            logger.info("Creating checkpoint: %s", gen_checkpoint_name)
            logger.info("Creating checkpoint: %s", discrim_checkpoint_name)
            if args.distributed:
                torch.save(discrim_model.module.state_dict(), discrim_checkpoint_name + '.pth')
                torch.save(gen_model.module.state_dict(), gen_checkpoint_name + '.pth')

            else:
                torch.save(discrim_model.state_dict(), discrim_checkpoint_name + '.pth')
                torch.save(gen_model.state_dict(), gen_checkpoint_name + '.pth')

            rm_old_checkpoints(gen_base, epoch)
            rm_old_checkpoints(discrim_base, epoch)


if __name__ == "__main__":
    train()

