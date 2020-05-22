import logging
import time
import os
from argparse import ArgumentParser
from collections import namedtuple
import baseline
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, write_json, revlut, print_table
import glob
from baseline.pytorch.embeddings import *
import baseline.embeddings
from eight_mile.optz import *
from eight_mile.pytorch.optz import *
from eight_mile.pytorch.layers import Average, checkpoint_for, rm_old_checkpoints
logger = logging.getLogger(__file__)
from baseline.pytorch.lm import TransformerMaskedLanguageModel
from eight_mile.pytorch.layers import TransformerEncoderStack, EmbeddingsStack, subsequent_mask
from transformer_utils import MultiFileDatasetReader, TransformerDiscriminator, find_latest_checkpoint, TensorWordDatasetReader, load_data_caching

"""Pre-train an discriminator Transformer model in PyTorch

This file uses Baseline to train a Transformer-based discriminative model
model, similar to (https://openreview.net/pdf?id=r1xMH1BtvB)
"""
LAMBDA = 50
Row = namedtuple('Row', 'original reconstructed guess')


def print_batch(index2word, labels, recon_labels, logits):
    j = 0

    for orig, recon, guess in zip(labels, recon_labels, logits):
        rows = []
        orig = orig.tolist()
        recon = recon.tolist()
        guess = guess.squeeze()
        guess = (guess > 0.5).tolist()

        for i in range(10):
            rows.append(Row(original=index2word[orig[i]], reconstructed=index2word[recon[i]], guess=str(guess[i])))
        print_table(rows)
        print('\n')
        if j == 3:
            return
        j += 1

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
    #return x_preds.argmax(axis=-1)
    B, T, V = x_preds.shape
    sample_dist = x_preds.exp().view(B * T, V)
    output = torch.multinomial(sample_dist, num_samples=1).view(B, T)
    #output = output.squeeze(0).item()
    return output


def get_accuracy(preds, true_or_fake, logits):
    flat_logits = logits.reshape(-1)
    nz_preds = preds.view(-1)[flat_logits != 0]
    nz_true_or_fake = true_or_fake.view(-1)[flat_logits != 0]

    preds_true = (nz_preds > 0.5).squeeze().to(nz_true_or_fake.dtype)
    num = torch.sum((nz_true_or_fake == preds_true).to(torch.float32))
    denom = nz_true_or_fake.nelement()
    return (num / denom).item()


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_file", type=str, help='Optional file path to use for train file')
    parser.add_argument("--valid_file", type=str, help='Optional file path to use for valid file')
    parser.add_argument("--streaming_load", type=str2bool, default=True,
                        help="whether loading data in a streaming way or loading all data into memory once")
    parser.add_argument("--gen_d_model", type=int, default=256, help="Model dimension (and embedding dsz)")
    parser.add_argument("--discrim_d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--gen_d_ff", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--discrim_d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--gen_d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--discrim_d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--gen_num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--discrim_num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--gen_num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--discrim_num_layers", type=int, default=8, help="Number of layers")
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
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from the latest checkpoint in a directory")
    parser.add_argument("--restart_tt", type=str, choices=['step', 'epoch'],
                        default='step',
                        help="Optional param for legacy checkpoints (step|epoch)")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--update_steps", type=int, default=100, help="The number of steps to take before saving a checkpoint")
    parser.add_argument("--print", type=str2bool, default=True, help="Print some output")
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

    if args.streaming_load:
        reader = MultiFileDatasetReader(args.nctx, args.subword_model_file, args.subword_vocab_file, args.pattern,
                                        reader_type="lang")
    else:
        reader = TensorWordDatasetReader(args.nctx, 'bpe', args.subword_model_file, args.subword_vocab_file)

    # This just return the vocab from the BPE vectorizer
    vocab = reader.build_vocab([])
    gen_embed = baseline.embeddings.load_embeddings('x', dsz=args.gen_d_model, known_vocab=vocab['x'],
                                                    embed_type=args.embed_type)
    vocabs = gen_embed['vocab']
    index2word = revlut(vocabs)
    discrim_embed = baseline.embeddings.load_embeddings('x', dsz=args.discrim_d_model, known_vocab=vocab['x'],
                                                        embed_type=args.embed_type)

    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(args.basedir, 'vocabs.json'))
    gen_embeddings = {'x': gen_embed['embeddings']}
    discrim_embeddings = {'x': discrim_embed['embeddings']}
    logger.info("Loaded embeddings")

    if args.streaming_load:
        train_set = reader.load(args.train_file, vocabs)
        valid_set = reader.load(args.valid_file, vocabs)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_train_workers)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_valid_workers)
        steps_per_epoch = len(train_loader) // (args.batch_size*num_gpus)
    else:
        dataset = {'train_file': args.train_file, 'valid_file': args.valid_file}
        train_set = load_data_caching('bpe', reader, dataset, 'train_file', {'x': vocabs}, True, logger)
        valid_set = load_data_caching('bpe', reader, dataset, 'valid_file', {'x': vocabs}, True, logger)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None
        train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size, shuffle=(not args.distributed))
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
        steps_per_epoch = len(train_loader)

    logger.info("Loaded datasets")
    logger.info("Using embedding type [%s]", args.embed_type)

    mask_value = vocabs.get("[MASK]", vocabs.get("<MASK>", -1))
    if mask_value == -1:
        logger.error("We could not find a suitable masking token in the vocab")
        return
    os.makedirs(args.basedir, exist_ok=True)
    vocab_size = len(vocabs)

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

    update_on = steps_per_epoch // args.update_steps
    report_on = update_on // 10
    logger.info(f"Steps per epoch per GPU: {steps_per_epoch}. Saving checkpoint every {update_on} steps.")
    cosine_decay = CosineDecaySchedulerPyTorch(steps_per_epoch * args.epochs, lr=args.lr)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, cosine_decay, lr=args.lr)

    global_step = 0
    start_epoch = 0
    if args.restart_from:
        if not os.path.isdir(args.restart_from):
            raise Exception(f"Cannot restart from {args.restart_from}, directory not found")
        tick_type = args.restart_tt
        discrim_latest = find_latest_checkpoint(args.restart_from, wildcard=f'checkpoint-discrim-{tick_type}')
        step_num = int(discrim_latest.split("-")[-1])
        gen_latest = find_latest_checkpoint(args.restart_from, wildcard=f'checkpoint-gen-{tick_type}')
        discrim_model.load_state_dict(torch.load(discrim_latest))
        gen_model.load_state_dict(torch.load(gen_latest))
        if tick_type == 'step':
            start_epoch = step_num // steps_per_epoch
            global_step = step_num
        else:
            start_epoch = step_num
            global_step = steps_per_epoch * start_epoch

    parameters = list(discrim_model.parameters()) + list(gen_model.parameters())
    optz = OptimizerManager(parameters, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
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
        avg_train_loss = Average('average5_train_loss')
        metrics = {}
        optz.zero_grad()
        start = time.time()
        for i, batch in enumerate(train_loader):
            steps += 1
            x, y = batch
            noised_x = x.to(args.device)
            # We are going to mask inplace and that will leave us with <PAD> anywhere that isnt MLM
            labels = y.to(args.device, copy=True)
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
            logits = gen_model({'x': noised_x}, None)[0]
            gen_loss_step = gen_loss_fn(logits.transpose(0, 1).contiguous(), labels)
            avg_gen_loss.update(gen_loss_step.item())
            # Re-read labels from device, this clears the masked <PAD>
            labels = y.to(args.device)

            # The logits needs to be replaced with either argmax or sampling.  Which?
            recon_labels = best_from(logits)
            recon_labels[~masked_indices] = labels[~masked_indices]
            true_or_fake = (recon_labels == labels).to(torch.float32).view(-1)
            logits = discrim_model({'x': recon_labels})
            discrim_loss_step = discrim_loss_fn(logits.view(-1), true_or_fake.view(-1))

            total_loss_step = gen_loss_step + LAMBDA * discrim_loss_step
            total_loss_step.backward()
            avg_discrim_loss.update(discrim_loss_step.item())
            avg_train_loss.update(total_loss_step.item())
            avg_discrim_acc.update(get_accuracy(logits, true_or_fake, labels))
            torch.nn.utils.clip_grad_norm_(parameters, args.clip)
            optz.step()
            optz.zero_grad()
            if (i + 1) % report_on == 0:
                logging.info('Loss g=%f, d=%f total=%f, Per token acc=%f', avg_gen_loss.avg, avg_discrim_loss.avg, avg_train_loss.avg, avg_discrim_acc.avg)
                if args.print:
                    print_batch(index2word, labels, recon_labels, logits)

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
        metrics['average_train_loss'] = avg_train_loss.avg

        avg_valid_gen_loss = Average('average_valid_gen_loss')
        avg_valid_discrim_loss = Average('average_valid_discrim_loss')
        avg_valid_discrim_acc = Average('average_valid_discrim_acc')
        avg_valid_loss = Average('average_valid_loss')
        start = time.time()
        gen_model.eval()
        discrim_model.eval()
        for batch in valid_loader:
            with torch.no_grad():
                x, y = batch
                noised_x = x.to(args.device)
                labels = y.to(args.device, copy=True)
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
                logits = gen_model({'x': noised_x}, None)[0]

                gen_loss_step = gen_loss_fn(logits.transpose(0, 1).contiguous(), labels)
                avg_valid_gen_loss.update(gen_loss_step.item())

                labels = y.to(args.device)
                recon_labels = best_from(logits)
                recon_labels[~masked_indices] = labels[~masked_indices]
                true_or_fake = (recon_labels == labels).to(torch.float32).view(-1)
                logits = discrim_model({'x': recon_labels})
                discrim_loss_step = discrim_loss_fn(logits.view(-1), true_or_fake.view(-1))
                avg_valid_discrim_acc.update(get_accuracy(logits, true_or_fake, labels))
                avg_valid_discrim_loss.update(discrim_loss_step.item())
                total_loss_step = gen_loss_step + LAMBDA * discrim_loss_step
                avg_valid_loss.update(total_loss_step.item())
        elapsed = (time.time() - start)/60
        metrics['valid_elapsed_min'] = elapsed
        metrics['average_valid_gen_loss'] = avg_valid_gen_loss.avg
        metrics['average_valid_discrim_loss'] = avg_valid_discrim_loss.avg
        metrics['average_valid_discrim_per_token_accuracy'] = avg_valid_discrim_acc.avg
        metrics['average_valid_loss'] = avg_valid_loss.avg
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

