import logging
import time
import os
from argparse import ArgumentParser
from collections import namedtuple
import baseline
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.utils import Timer, str2bool, write_json, revlut, print_table
import glob
from baseline.pytorch.embeddings import *
import baseline.embeddings
from eight_mile.optz import *
from eight_mile.pytorch.optz import *
from eight_mile.utils import Average, get_num_gpus_multiworker
from eight_mile.pytorch.layers import checkpoint_for, rm_old_checkpoints, save_checkpoint, init_distributed
logger = logging.getLogger(__file__)
from baseline.pytorch.lm import TransformerMaskedLanguageModel
from eight_mile.pytorch.layers import TransformerDiscriminator
from mead.api_examples.transformer_utils import MultiFileDatasetReader, find_latest_checkpoint, get_lr_decay, on_demand_mlm_masking

"""Pre-train an discriminator Transformer model in PyTorch

This file uses Baseline to train a Transformer-based discriminative model
model, similar to (https://openreview.net/pdf?id=r1xMH1BtvB)

If the data has been previously preprocessed, lets assume that its been processed to MLM format, where
`x` is the noised version and `y` is padded everywhere that there is no noising.  This means we should not run
`on_demand_masking()` where normally we would.

If the data is in the default format (`lang`), we assume that it produces identical values for `x` and `y`
and during the `on_demand_masking()` we mask x and preserve y values where x is masked, setting the rest to <PAD>
"""
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


def gen_vs_discrim(x, y, device, gen_model, gen_loss_fn, discrim_model, discrim_loss_fn, mask_value, vocab_size, index2word, print_output, do_on_demand_masking):
    inputs = x.to(device)
    # We are going to mask inplace and that will leave us with <PAD> anywhere that isnt MLM
    # If x and y have been preprocessed, then x has already been noised any y is reconstructed in the masked regions only
    # , so nothing to do
    # If it hasnt been preprocessed, then we need to do that on demand
    labels = y.to(device, copy=True)
    if do_on_demand_masking:
        noised_x, labels, masked_indices = on_demand_mlm_masking(inputs, labels, mask_value, vocab_size)
    else:
        noised_x = inputs
        masked_indices = (noised_x == mask_value)

    logits = gen_model({'x': noised_x}, None)[0]
    gen_loss_step = gen_loss_fn(logits.transpose(0, 1).contiguous(), labels.transpose(0, 1).contiguous())

    # Re-read labels from device, this clears the masked <PAD>
    #labels = y.to(device)
    # The logits needs to be replaced with either argmax or sampling.  Which?
    recon_labels = best_from(logits)
    recon_labels[~masked_indices] = noised_x[~masked_indices]
    true_or_fake = ~masked_indices
    true_or_fake[recon_labels == labels] = True
    true_or_fake = true_or_fake.to(torch.float32).view(-1).to(noised_x.device)
    logits = discrim_model({'x': recon_labels})
    discrim_loss_step = discrim_loss_fn(logits.view(-1), true_or_fake.view(-1))
    acc = get_accuracy(logits, true_or_fake)
    if print_output:
        print_batch(index2word, y, recon_labels, logits)
    return gen_loss_step, discrim_loss_step, acc


def best_from(x_preds):
    #return x_preds.argmax(axis=-1)
    B, T, V = x_preds.shape
    sample_dist = x_preds.exp().view(B * T, V)
    output = torch.multinomial(sample_dist, num_samples=1).view(B, T)
    return output


def get_accuracy(preds, true_or_fake):
    nz_preds = preds.view(-1)
    nz_true_or_fake = true_or_fake.view(-1)

    preds_true = (nz_preds > 0.5).squeeze().to(nz_true_or_fake.dtype)
    num = torch.sum((nz_true_or_fake == preds_true).to(torch.float32))
    denom = nz_true_or_fake.nelement()
    return (num / denom).item()


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_file", type=str, help='Optional file path to use for train file')
    parser.add_argument("--valid_file", type=str, help='Optional file path to use for valid file')
    parser.add_argument("--preprocessed", type=str2bool, default=True, help="Has the data already been preprocessed?")

    parser.add_argument("--gen_d_model", type=int, default=256, help="Model dimension (and embedding dsz)")
    parser.add_argument("--gen_d_ff", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--gen_d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--gen_num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--gen_num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--gen_dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument('--gen_rpr_k',
                        help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')

    parser.add_argument("--discrim_d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--discrim_d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--discrim_d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--discrim_num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--discrim_num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--discrim_dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument('--discrim_rpr_k', help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')

    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--nctx", type=int, default=256, help="Max context length (for both encoder and decoder)")
    parser.add_argument("--embed_type", type=str, default='default',
                        choices=["default", "positional", "learned-positional"],
                        help="register label of the embeddings")
    parser.add_argument("--pattern", default='*.json', help="Glob pattern for files, defaults to *.json if preprocessed, *.txt otherwise")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--dataset_key", default="reddit",
                        help="dataset key for basedir")
    parser.add_argument("--subword_model_file", type=str, required=True)
    parser.add_argument("--subword_vocab_file", type=str, required=True)
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_decay_steps", type=int, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of lr scheduler")
    parser.add_argument("--lr_alpha", type=float, help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--optim", default="adam", type=str, help="Optimizer to use (defaults to adam)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--gen_loss_scale", type=float, default=50.0, help="Scaling for loss function")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=32, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from the latest checkpoint in a directory")
    parser.add_argument("--restart_tt", type=str, choices=['step', 'epoch'],
                        default='step',
                        help="Optional param for legacy checkpoints (step|epoch)")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--saves_per_epoch", type=int, default=100, help="The number of checkpoints to save per epoch")
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
    num_gpus = get_num_gpus_multiworker()
    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

    if args.distributed:
        args.device, args.local_rank = init_distributed(args.local_rank)

    if not args.preprocessed:
        reader_type = "lang"
        args.pattern = "*.txt"
    else:
        reader_type = "preprocessed"
    reader = MultiFileDatasetReader(args.nctx, args.subword_model_file, args.subword_vocab_file, args.pattern,
                                    reader_type=reader_type)
    #  just return the vocab from the BPE vectorizer
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

    train_set = reader.load(args.train_file, vocabs)
    valid_set = reader.load(args.valid_file, vocabs)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_train_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    train_steps_per_epoch = len(train_loader) // (args.batch_size*num_gpus)
    valid_steps_per_epoch = len(valid_loader) // args.batch_size
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
    elif len(args.gen_rpr_k) == 1:
        gen_rpr_k = args.gen_rpr_k[0]
    else:
        gen_rpr_k = args.gen_rpr_k

    if len(args.gen_rpr_k) == 0 or args.discrim_rpr_k[0] < 1:
        discrim_rpr_k = None
    elif len(args.discrim_rpr_k) == 1:
        discrim_rpr_k = args.discrim_rpr_k[0]
    else:
        discrim_rpr_k = args.discrim_rpr_k

    gen_model = TransformerMaskedLanguageModel.create(gen_embeddings, hsz=args.gen_d_model, d_ff=args.gen_d_ff,
                                                      tie_weights=True, dropout=args.gen_dropout,
                                                      num_heads=args.gen_num_heads, layers=args.gen_num_layers,
                                                      rpr_k=gen_rpr_k, d_k=args.gen_d_k, src_keys=['x'], tgt_key='x')
    discrim_model = TransformerDiscriminator(discrim_embeddings, d_model=args.discrim_d_model, d_ff=args.discrim_d_ff,
                                             dropout=args.discrim_dropout, num_heads=args.discrim_num_heads,
                                             layers=args.discrim_num_layers,
                                             activation='gelu', layer_norm_eps=1.0e-12,
                                             rpr_k=discrim_rpr_k, d_k=args.discrim_d_k)
    gen_model.to(args.device)
    gen_loss_fn = gen_model.create_loss()

    discrim_model.to(args.device)
    discrim_loss_fn = discrim_model.create_loss()
    logger.info("Loaded model and loss")

    update_on = train_steps_per_epoch // args.saves_per_epoch
    report_on = update_on // 10
    logger.info(f"Steps per epoch per GPU: {train_steps_per_epoch}. Saving checkpoint every {update_on} steps.")
    lr_decay = get_lr_decay(args.lr_scheduler, args.lr, train_steps_per_epoch, args.epochs, logger,
                            decay_steps=args.lr_decay_steps, decay_rate=args.lr_decay_rate, alpha=args.lr_alpha)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, lr_decay, lr=args.lr)

    global_step = 0
    start_epoch = 0
    if args.restart_from:
        if not os.path.isdir(args.restart_from):
            raise Exception(f"Cannot restart from {args.restart_from}, directory not found")
        tick_type = args.restart_tt
        discrim_latest, step_num = find_latest_checkpoint(args.restart_from, wildcard=f'checkpoint-discrim-{tick_type}')
        gen_latest, _ = find_latest_checkpoint(args.restart_from, wildcard=f'checkpoint-gen-{tick_type}')
        discrim_model.load_state_dict(torch.load(discrim_latest))
        gen_model.load_state_dict(torch.load(gen_latest))
        if tick_type == 'step':
            start_epoch = step_num // train_steps_per_epoch
            global_step = step_num
        else:
            start_epoch = step_num
            global_step = train_steps_per_epoch * start_epoch

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
    do_on_demand_masking = not args.preprocessed
    if do_on_demand_masking:
        logger.info(f"On-demand masking is turned on")

    timer = Timer()

    for epoch in range(start_epoch, args.epochs):
        gen_model.train()
        discrim_model.train()
        avg_gen_loss = Average('average_train_gen_loss')
        avg_discrim_loss = Average('average_train_discrim_loss')
        avg_discrim_acc = Average('average_train_discrim_acc')
        avg_train_loss = Average('average5_train_loss')
        metrics = {}
        optz.zero_grad()
        timer.start()
        print(f'Starting epoch {epoch + 1}')
        train_iter = iter(train_loader)
        valid_iter = iter(valid_loader)

        for i in range(train_steps_per_epoch):
            steps += 1
            x, y = next(train_iter)
            do_report = (i + 1) % report_on == 0 and args.print
            gen_loss_step, discrim_loss_step, acc = gen_vs_discrim(x, y, args.device, gen_model, gen_loss_fn,
                                                                   discrim_model, discrim_loss_fn, mask_value,
                                                                   vocab_size, index2word, do_report, do_on_demand_masking)
            avg_gen_loss.update(gen_loss_step.item())
            total_loss_step = gen_loss_step + args.gen_loss_scale * discrim_loss_step
            total_loss_step.backward()
            avg_discrim_loss.update(discrim_loss_step.item())
            avg_train_loss.update(total_loss_step.item())
            avg_discrim_acc.update(acc)
            torch.nn.utils.clip_grad_norm_(parameters, args.clip)
            optz.step()
            optz.zero_grad()
            if (i + 1) % report_on == 0:
                logging.info('Loss g=%f, d=%f total=%f, Per token acc=%f', avg_gen_loss.avg, avg_discrim_loss.avg, avg_train_loss.avg, avg_discrim_acc.avg)

            if (i + 1) % update_on == 0 and args.local_rank < 1:
                elapsed = timer.elapsed(True)
                logging.info('elapsed time this epoch %d min', elapsed)
                logging.info('elapsed step time %f steps/min', i/elapsed)
                logging.info('LR: %f',  optz.current_lr)
                save_checkpoint(gen_model, gen_base, steps, tick_type='step')
                save_checkpoint(discrim_model, discrim_base, steps, tick_type='step')

        # How much time elapsed in minutes
        elapsed = timer.elapsed(True)
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_gen_loss'] = avg_gen_loss.avg
        metrics['average_train_discrim_loss'] = avg_discrim_loss.avg
        metrics['average_train_discrim_per_token_accuracy'] = avg_discrim_acc.avg
        metrics['average_train_loss'] = avg_train_loss.avg

        if args.local_rank < 1:
            avg_valid_gen_loss = Average('average_valid_gen_loss')
            avg_valid_discrim_loss = Average('average_valid_discrim_loss')
            avg_valid_discrim_acc = Average('average_valid_discrim_acc')
            avg_valid_loss = Average('average_valid_loss')
            timer.start()
            gen_model.eval()
            discrim_model.eval()
            for i in range(valid_steps_per_epoch):
                with torch.no_grad():
                    x, y = next(valid_iter)
                    do_report = (i + 1) % report_on == 0 and args.print
                    gen_loss_step, discrim_loss_step, acc = gen_vs_discrim(x, y, args.device, gen_model, gen_loss_fn,
                                                                           discrim_model, discrim_loss_fn, mask_value,
                                                                           vocab_size, index2word, do_report, do_on_demand_masking)
                    avg_valid_gen_loss.update(gen_loss_step.item())
                    avg_valid_discrim_acc.update(acc)
                    avg_valid_discrim_loss.update(discrim_loss_step.item())
                    total_loss_step = gen_loss_step + args.gen_loss_scale * discrim_loss_step
                    avg_valid_loss.update(total_loss_step.item())
            elapsed = timer.elapsed(True)
            metrics['valid_elapsed_min'] = elapsed
            metrics['average_valid_gen_loss'] = avg_valid_gen_loss.avg
            metrics['average_valid_discrim_loss'] = avg_valid_discrim_loss.avg
            metrics['average_valid_discrim_per_token_accuracy'] = avg_valid_discrim_acc.avg
            metrics['average_valid_loss'] = avg_valid_loss.avg
            logger.info(metrics)
            save_checkpoint(discrim_model, discrim_base, epoch, tick_type='epoch', save_npz=True)
            save_checkpoint(gen_model, gen_base, epoch, tick_type='epoch', save_npz=True)


if __name__ == "__main__":
    train()

