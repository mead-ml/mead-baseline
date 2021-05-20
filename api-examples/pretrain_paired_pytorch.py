import logging
import time
import os
from argparse import ArgumentParser
import baseline
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, write_json, Average, Timer, get_num_gpus_multiworker
from baseline.pytorch.embeddings import *
import baseline.embeddings
from eight_mile.optz import *
from eight_mile.pytorch.serialize import load_seq2seq_enc_from_tlm_npz, load_transformer_seq2seq_npz, load_transformer_de_npz
from eight_mile.pytorch.layers import (
    save_checkpoint, init_distributed,
    PairedModel,
    TransformerBoWPairedModel,
)
from baseline.pytorch.seq2seq.model import TiedEmbeddingsSeq2SeqModel
from eight_mile.pytorch.optz import *
from transformer_utils import (
    MultiFileDatasetReader,
    get_lr_decay,
)


logger = logging.getLogger(__file__)


"""Pre-train a paired model in PyTorch

This file uses Baseline to train a Transformer model using fastBPE with query-response input using either
  - Dual-encoder model (https://arxiv.org/pdf/1911.03688.pdf) with AllLoss or TripletLoss
  - Seq2Seq model with response generation
  
"""
def create_model(embeddings, d_model, d_ff, dropout, num_heads, num_layers, model_type, rpr_k, d_k, reduction_d_k,
                 stacking_layers, ff_pdrop, windowed_ra, reduction_type, layer_drop, logger):
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
               "layer_drop": layer_drop,
               "rpr_k": rpr_k}
        model = TiedEmbeddingsSeq2SeqModel({'x': embeddings}, None, **hps)
    elif model_type == 'transformer-bow':
        model = TransformerBoWPairedModel(embeddings, d_model, d_ff, dropout, num_heads, num_layers, rpr_k=rpr_k, d_k=d_k,
                                          reduction_d_k=reduction_d_k, stacking_layers=stacking_layers, ffn_pdrop=ff_pdrop, windowed_ra=windowed_ra,
                                          reduction_type_1=reduction_type, freeze_encoders=True)
    else:
        model = PairedModel(embeddings, d_model, d_ff, dropout, num_heads, num_layers, rpr_k=rpr_k, d_k=d_k,
                            reduction_d_k=reduction_d_k, stacking_layers=stacking_layers, ffn_pdrop=ff_pdrop,
                            windowed_ra=windowed_ra, reduction_type=reduction_type, freeze_encoders=True)

    logger.info(model)
    return model


def run_step_dual(x, y, model, loss_function, device, distributed):
    inputs = x.to(device)
    labels = y.to(device)
    loss = loss_function(inputs, labels)
    return loss


def run_step_s2s(x, y, model, loss_function, device, distributed):
    x_lengths = torch.sum(x != Offsets.PAD, 1)
    y_lengths = torch.sum(y != Offsets.PAD, 1)
    if distributed:
        inputs = model.module.make_input({'x': x, 'x_lengths': x_lengths, 'tgt': y, 'tgt_lengths': y_lengths})
        pred = model(inputs)
    else:
        inputs = model.make_input({'x': x, 'x_lengths': x_lengths, 'tgt': y, 'tgt_lengths': y_lengths})
        pred = model(inputs)
    loss = loss_function(pred, inputs['tgt'])
    return loss


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_file", type=str, required=True, help='File path to use for train file')
    parser.add_argument("--valid_file", type=str, required=True, help='File path to use for valid file')
    parser.add_argument("--dataset_key", default="paired",
                        help="dataset key for basedir")
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
    parser.add_argument("--tgt_nctx", type=int, help="Max output length, default to args.nctx")
    parser.add_argument("--file_type", default='json', help="Suffix for data")
    parser.add_argument("--record_keys", default=['x', 'y'], nargs='+')
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--subword_model_file", type=str, help="The BPE model file", required=True)
    parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_decay_steps", type=int, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of lr scheduler")
    parser.add_argument("--lr_alpha", type=float, help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=32, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--restart_tt", type=str, help="Optional param for legacy checkpoints (step|epoch)")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--saves_per_epoch", type=int, default=10, help="The number of checkpoints to save per epoch")
    parser.add_argument("--reduction_d_k", type=int, default=64, help="Dimensions of Key and Query in the single headed"
                                                                      "reduction layers")
    parser.add_argument("--reduction_type", type=str, default="2ha", help="If using a dual encoder, specifies the reduction type")
    parser.add_argument("--unfreeze_after_step", default=0, type=int, help="Unfreeze encoders after step, ignored if we dont have a checkpoint")
    parser.add_argument("--stacking_layers", type=int, nargs='+', default=[],
                        help="Hidden sizes of the dense stack (ff2 from the convert paper)")
    parser.add_argument("--layer_drop", type=float, default=0.0, help="LayerDrop to apply")
    parser.add_argument("--ff_pdrop", type=float, default=0.1, help="Dropout in the dense stack")

    parser.add_argument("--reader_type", type=str, default='preprocessed', choices=['ntp', 'nsp', 'preprocessed', 'tfrecord'])
    parser.add_argument("--model_type", default="dual-encoder", choices=["dual-encoder", "encoder-decoder", "transformer-bow"])
    parser.add_argument("--src_begin_tok", type=str, nargs='+', default=[])
    parser.add_argument("--src_end_tok", type=str, nargs='+', default=['<EOS>'])
    parser.add_argument("--tgt_begin_tok", type=str, nargs='+', default=['<GO>'])
    parser.add_argument("--tgt_end_tok", type=str, nargs='+', default=['<EOS>'])
    parser.add_argument('--lower', type=baseline.str2bool, default=False)
    parser.add_argument("--loss", type=str, default='symmetric', choices=['triplet', 'all', 'all_mean', 'contrastive', 'symmetric'])
    parser.add_argument("--learn_temp", type=str2bool, default=True,
                        help="If 'constrastive' or 'symmetric' loss, should we learn the temperature scaling")
    parser.add_argument("--init_temp", type=float, help="Initialize the temperature for 'contrastive' or 'symmetric' loss")
    parser.add_argument('--rpr_k',
                        help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')
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
    parser.add_argument("--save_npz", type=str2bool, default=False, help="Whether save npz checkpoint")

    args = parser.parse_args()

    if args.basedir is None: 
        args.basedir = '{}-{}-paired-{}-bpe-{}'.format(args.model_type, args.reader_type, args.dataset_key, os.getpid())
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    num_gpus = get_num_gpus_multiworker()
    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

    if args.distributed:
        args.device, updated_local_rank = init_distributed(args.local_rank)
        args.local_rank = updated_local_rank

    if not args.tgt_nctx:
        args.tgt_nctx = args.nctx
    reader = MultiFileDatasetReader(args.nctx, args.tgt_nctx, args.src_begin_tok, args.src_end_tok, args.tgt_begin_tok,
                                    args.tgt_end_tok, args.subword_model_file, args.subword_vocab_file,
                                    args.file_type, reader_type=args.reader_type, record_keys=args.record_keys,
                                    lower=args.lower)

    vocab = reader.build_vocab()
    # If we are not using chars, then use 'x' for both input and output
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, known_vocab=vocab['x'],
                                                       preserve_vocab_indices=True,
                                                       embed_type=args.embed_type)
    vocabs = preproc_data['vocab']

    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(args.basedir, 'vocabs.json'))
    embeddings = preproc_data['embeddings']
    logger.info("Loaded embeddings")

    train_set = reader.load(args.train_file, vocabs)
    valid_set = reader.load(args.valid_file, vocabs, distribute=False, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_train_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    logger.info("Loaded datasets")
    logger.info("Using embedding type [%s]", args.embed_type)

    if len(args.rpr_k) == 0 or args.rpr_k[0] < 1:
        rpr_k = None
    elif len(args.rpr_k) == 1:
        rpr_k = args.rpr_k[0]
    else:
        rpr_k = args.rpr_k

    model = create_model(embeddings, d_model=args.d_model, d_ff=args.d_ff, dropout=args.dropout,
                         num_heads=args.num_heads, num_layers=args.num_layers,
                         model_type=args.model_type, rpr_k=rpr_k, d_k=args.d_k, reduction_d_k=args.reduction_d_k,
                         stacking_layers=args.stacking_layers, ff_pdrop=args.ff_pdrop, windowed_ra=args.windowed_ra,
                         reduction_type=args.reduction_type,
                         layer_drop=args.layer_drop,
                         logger=logger)

    model.to(args.device)
    if args.model_type == 'encoder-decoder':
        run_step = run_step_s2s
    else:
        run_step = run_step_dual
        logger.info(f"Creating {args.loss}, init temperature: {args.init_temp}, learnable: {args.learn_temp}")
    loss_function = model.create_loss(loss_type=args.loss, init_temp=args.init_temp, learn_temp=args.learn_temp)
    loss_function.to(args.device)

    logger.info("Created model and loss")

    steps_per_epoch = len(train_loader) // num_gpus
    valid_steps = len(valid_loader)
    update_on = steps_per_epoch // args.saves_per_epoch
    report_on = max(10, update_on) // 10
    logger.info(f"Steps per epoch per GPU: {steps_per_epoch}. Saving checkpoint every {update_on} steps.")
    lr_decay = get_lr_decay(args.lr_scheduler, args.lr, steps_per_epoch, args.epochs, logger,
                            decay_steps=args.lr_decay_steps, decay_rate=args.lr_decay_rate, alpha=args.lr_alpha)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, lr_decay, lr=args.lr)

    global_step = 0
    start_epoch = 0

    if args.restart_from:

        if args.unfreeze_after_step > 0 and args.model_type == "dual-encoder":
            logger.info(f"Encoders will be frozen until step %d", args.unfreeze_after_step)
        global_step, start_epoch = reload_from_checkpoint(args.model_type, args.restart_from, args.restart_tt, model, steps_per_epoch)
        logger.info("Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d, epoch=%d",
                    args.restart_from, global_step, start_epoch+1)

    target = model if args.model_type == 'encoder-decoder' else loss_function

    optimizer = OptimizerManager(target, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
    logger.info("Model has {:,} parameters".format(sum(p.numel() for p in target.parameters() if p.requires_grad)))
    # Prepare model for distributed training if needed
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.device], output_device=args.device)
        logger.info("Model located on %d", args.local_rank)

    model_base = os.path.join(args.basedir, 'checkpoint')
    steps = global_step
    timer = Timer()

    for epoch in range(start_epoch, args.epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}
        optimizer.zero_grad()
        timer.start()
        model.train()
        train_itr = iter(train_loader)
        for i in range(steps_per_epoch):
            batch = next(train_itr)

            if steps > args.unfreeze_after_step and hasattr(model, 'freeze') and model.freeze:
                logging.info("Unfreezing encoders at step %d", steps)
                model.freeze = False
            steps += 1

            x, y = batch
            loss = run_step(x, y, model, loss_function, args.device, args.distributed)
            loss.backward()
            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % report_on == 0:
                logging.info(avg_loss)
            if (i + 1) % update_on == 0 and args.local_rank < 1:
                elapsed = timer.elapsed(True)
                logging.info('elapsed time this epoch %d min', elapsed)
                logging.info('elapsed step time %f steps/min', i/elapsed)
                logging.info('LR: %f',  optimizer.current_lr)
                save_checkpoint(model, model_base, steps, tick_type='step', save_npz=args.save_npz)

        # How much time elapsed in minutes
        elapsed = timer.elapsed(True)
        train_avg_loss = avg_loss.avg
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_avg_loss
        if args.local_rank < 1:
            avg_valid_loss = Average('average_valid_loss')
            timer.start()
            model.eval()
            valid_itr = iter(valid_loader)
            for j in range(valid_steps):
                with torch.no_grad():
                    batch = next(valid_itr)
                    x, y = batch
                    loss = run_step(x, y, model, loss_function, args.device, args.distributed)
                avg_valid_loss.update(loss.item())

            valid_avg_loss = avg_valid_loss.avg

            elapsed = timer.elapsed(True)
            metrics['valid_elapsed_min'] = elapsed

            metrics['average_valid_loss'] = valid_avg_loss
            logger.info(metrics)
            save_checkpoint(model, model_base, epoch, tick_type='epoch', save_npz=args.save_npz)


def reload_from_checkpoint(model_type, restart_from, restart_tick_type, model, steps_per_epoch):
    if os.path.isdir(restart_from):
        restart_from, _ = find_latest_checkpoint(restart_from)
        print(f'Latest checkpoint: {restart_from}')
    vec = restart_from.split("-")
    step_num = int(vec[-1].split(".")[0])
    start_epoch = 0
    if restart_tick_type:
        tick_type = restart_tick_type
    else:
        tick_type = vec[-2]
    if restart_from.endswith('.npz'):
        # If its a seq2seq load either from a seq2seq or from a TLM encoder
        if model_type == 'encoder-decoder':
            try:
                load_transformer_seq2seq_npz(model, restart_from)
            except:
                print('Model file not recognized as seq2seq model, attempting to load as LM for encoder, reset step')
                load_seq2seq_enc_from_tlm_npz(model, restart_from)
                step_num = 0
                tick_type = 'ignore'
        else:
            try:
                load_transformer_de_npz(model, restart_from)
            # If its a dual-encoder, assuming we have model.transformer and model.embeddings, we can load directly
            # from a Transformer Language Model
            except:
                print('Model file not recognized as a dual encoder model, attempting to load as LM for encoder, reset step')
                load_tlm_npz(model, restart_from)
                step_num = 0
                tick_type = 'ignore'

    else:
        model.load_state_dict(torch.load(restart_from))
    if tick_type == 'epoch':
        start_epoch = step_num
        step_num = start_epoch * steps_per_epoch

    elif tick_type == 'step':
        start_epoch = step_num // steps_per_epoch
    else:
        logger.warning(f"The previous tick was {step_num} but command-line specifies to ignore, setting to 0")
        step_num = 0
        start_epoch = 0
    return step_num, start_epoch


if __name__ == "__main__":
    train()
