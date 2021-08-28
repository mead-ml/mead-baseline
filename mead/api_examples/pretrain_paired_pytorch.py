import logging
import time
import os
import sys
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
from mead.api_examples.transformer_utils import MultiFileDatasetReader, get_lr_decay


logger = logging.getLogger(__file__)


"""Pre-train a paired model in PyTorch

This file uses Baseline to train a Transformer model using fastBPE with query-response input using either
  - Dual-encoder model (https://arxiv.org/pdf/1911.03688.pdf) with AllLoss or TripletLoss
  - Seq2Seq model with response generation
  
"""
def create_model(embeddings, d_model, d_ff, dropout, num_heads, num_layers, model_type, rpr_k, d_k, reduction_d_k,
                 stacking_layers, ff_pdrop, windowed_ra, reduction_type, layer_drop, logger, layer_norms_after=False):
    if model_type == "encoder-decoder":
        logger.info("Creating tied encoder decoder model")
        if layer_norms_after:
            raise Exception("Unsupported option, require pre layer norm")
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
                                          reduction_type_1=reduction_type, freeze_encoders=True, layer_norms_after=layer_norms_after)
    else:
        model = PairedModel(embeddings, d_model, d_ff, dropout, num_heads, num_layers, rpr_k=rpr_k, d_k=d_k,
                            reduction_d_k=reduction_d_k, stacking_layers=stacking_layers, ffn_pdrop=ff_pdrop,
                            windowed_ra=windowed_ra, reduction_type=reduction_type, freeze_encoders=True, layer_norms_after=layer_norms_after)

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


def main():
    argv = sys.argv[1:]
    args = parse_args(argv)
    run(**vars(args))


def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_file", type=str, required=True, help='File path to use for train file')
    parser.add_argument("--valid_file", type=str, required=True, help='File path to use for valid file')
    parser.add_argument("--dataset_key", default="paired",
                        help="dataset key for basedir")
    parser.add_argument("--embed_type", type=str, default='default',
                        choices=["default", "positional", "learned-positional", "learned-positional-w-bias"],
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
    parser.add_argument("--subword_type", type=str, choices=["bpe", "wordpiece"], default="bpe")
    parser.add_argument("--subword_model_file", type=str, help="The BPE model file")
    parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--layer_norms_after", type=str2bool, default=False, help="Layer norms after (set True for BERT)")
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
    parser.add_argument("--reduction_type", type=str, default="2ha",
                        help="If using a dual encoder, specifies the reduction type")
    parser.add_argument("--unfreeze_after_step", default=0, type=int,
                        help="Unfreeze encoders after step, ignored if we dont have a checkpoint")
    parser.add_argument("--stacking_layers", type=int, nargs='+', default=[],
                        help="Hidden sizes of the dense stack (ff2 from the convert paper)")
    parser.add_argument("--layer_drop", type=float, default=0.0, help="LayerDrop to apply")
    parser.add_argument("--ff_pdrop", type=float, default=0.1, help="Dropout in the dense stack")
    parser.add_argument("--reader_type", type=str, default='preprocessed',
                        choices=['ntp', 'nsp', 'preprocessed', 'tfrecord'])
    parser.add_argument("--model_type", default="dual-encoder",
                        choices=["dual-encoder", "encoder-decoder", "transformer-bow"])
    parser.add_argument("--src_begin_tok", type=str, nargs='+', default=[])
    parser.add_argument("--src_end_tok", type=str, nargs='+', default=['<EOS>'])
    parser.add_argument("--tgt_begin_tok", type=str, nargs='+', default=['<GO>'])
    parser.add_argument("--tgt_end_tok", type=str, nargs='+', default=['<EOS>'])
    parser.add_argument('--lower', type=baseline.str2bool, default=False)
    parser.add_argument("--loss", type=str, default='symmetric',
                        choices=['triplet', 'all', 'all_mean', 'contrastive', 'symmetric'])
    parser.add_argument("--learn_temp", type=str2bool, default=True,
                        help="If 'constrastive' or 'symmetric' loss, should we learn the temperature scaling")
    parser.add_argument("--init_temp", type=float,
                        help="Initialize the temperature for 'contrastive' or 'symmetric' loss")
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
    parser.add_argument("--extra_tokens", help="What extra tokens should we use", nargs="+", default=["[CLS]", "[MASK]"])
    parser.add_argument("--save_npz", type=str2bool, default=False, help="Whether save npz checkpoint")
    args = parser.parse_args(argv)
    return args


def run(basedir=None, train_file=None, valid_file=None, dataset_key='paired', embed_type='default',
        d_model=512, d_ff=2048, d_k=None, num_heads=8, num_layers=8, windowed_ra=False, num_train_workers=4,
        nctx=256, tgt_nctx=None, file_type='json', record_keys=['x', 'y'], batch_size=256,
        subword_model_file=None, subword_vocab_file=None, dropout=0.1, lr_scheduler='cosine',
        lr_decay_steps=None, lr_decay_rate=None, lr_alpha=None, optim='adamw', lr=4.0e-4, clip=1.0,
        weight_decay=1.0e-2, epochs=32, restart_from=None, restart_tt=None, warmup_steps=10000, saves_per_epoch=10,
        reduction_d_k=64, reduction_type='2ha', unfreeze_after_step=0, stacking_layers=[], layer_drop=0.0, ff_pdrop=0.1,
        reader_type='preprocessed', model_type='dual-encoder', src_begin_tok=[], src_end_tok=['<EOS>'],
        tgt_begin_tok=['<GO>'], tgt_end_tok=['<EOS>'], lower=False, loss='symmetric', learn_temp=True, init_temp=None,
        rpr_k=[8], device='cuda', distributed=False, local_rank=-1, save_npz=False, layer_norms_after=False,
        extra_tokens=["[CLS]", "[MASK]"], subword_type='bpe', **kwargs):
    if basedir is None:
        basedir = '{}-{}-paired-{}-bpe-{}'.format(model_type, reader_type, dataset_key, os.getpid())
    logging.basicConfig(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    num_gpus = get_num_gpus_multiworker()
    distributed = distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")
    if distributed:
        device, updated_local_rank = init_distributed(local_rank)
        local_rank = updated_local_rank
    if not tgt_nctx:
        tgt_nctx = nctx
    reader = MultiFileDatasetReader(nctx, tgt_nctx, src_begin_tok, src_end_tok, tgt_begin_tok,
                                    tgt_end_tok, subword_model_file, subword_vocab_file,
                                    file_type, reader_type=reader_type, record_keys=record_keys,
                                    lower=lower, extra_tokens=extra_tokens, subword_type=subword_type)
    vocab = reader.build_vocab()
    # If we are not using chars, then use 'x' for both input and output
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=d_model, known_vocab=vocab['x'],
                                                       preserve_vocab_indices=True,
                                                       embed_type=embed_type)
    vocabs = preproc_data['vocab']
    os.makedirs(basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(basedir, 'vocabs.json'))
    embeddings = preproc_data['embeddings']
    logger.info("Loaded embeddings")
    train_set = reader.load(train_file, vocabs)
    valid_set = reader.load(valid_file, vocabs, distribute=False, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_train_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    logger.info("Loaded datasets")
    logger.info("Using embedding type [%s]", embed_type)
    if len(rpr_k) == 0 or rpr_k[0] < 1:
        rpr_k = None
    elif len(rpr_k) == 1:
        rpr_k = rpr_k[0]
    else:
        rpr_k = rpr_k
    model = create_model(embeddings, d_model=d_model, d_ff=d_ff, dropout=dropout,
                         num_heads=num_heads, num_layers=num_layers,
                         model_type=model_type, rpr_k=rpr_k, d_k=d_k, reduction_d_k=reduction_d_k,
                         stacking_layers=stacking_layers, ff_pdrop=ff_pdrop, windowed_ra=windowed_ra,
                         reduction_type=reduction_type,
                         layer_drop=layer_drop,
                         layer_norms_after=layer_norms_after,
                         logger=logger)
    model.to(device)
    if model_type == 'encoder-decoder':
        run_step = run_step_s2s
    else:
        run_step = run_step_dual
        logger.info(f"Creating {loss}, init temperature: {init_temp}, learnable: {learn_temp}")
    loss_function = model.create_loss(loss_type=loss, init_temp=init_temp, learn_temp=learn_temp)
    loss_function.to(device)
    logger.info("Created model and loss")
    steps_per_epoch = len(train_loader) // num_gpus
    valid_steps = len(valid_loader)
    update_on = steps_per_epoch // saves_per_epoch
    report_on = max(10, update_on) // 10
    logger.info(f"Steps per epoch per GPU: {steps_per_epoch}. Saving checkpoint every {update_on} steps.")
    lr_decay = get_lr_decay(lr_scheduler, lr, steps_per_epoch, epochs, logger,
                            decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, alpha=lr_alpha)
    linear_warmup = WarmupLinearSchedulerPyTorch(warmup_steps, lr=lr)
    lr_sched = CompositeLRScheduler(linear_warmup, lr_decay, lr=lr)
    global_step = 0
    start_epoch = 0
    if restart_from:

        if unfreeze_after_step > 0 and model_type == "dual-encoder":
            logger.info(f"Encoders will be frozen until step %d", unfreeze_after_step)
        global_step, start_epoch = reload_from_checkpoint(model_type, restart_from, restart_tt, model,
                                                          steps_per_epoch)
        logger.info("Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d, epoch=%d",
                    restart_from, global_step, start_epoch + 1)
    target = model if model_type == 'encoder-decoder' else loss_function
    optimizer = OptimizerManager(target, global_step, optim=optim, lr=lr, lr_function=lr_sched,
                                 weight_decay=weight_decay)
    logger.info("Model has {:,} parameters".format(sum(p.numel() for p in target.parameters() if p.requires_grad)))
    # Prepare model for distributed training if needed
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
        logger.info("Model located on %d", local_rank)
    model_base = os.path.join(basedir, 'checkpoint')
    steps = global_step
    timer = Timer()
    for epoch in range(start_epoch, epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}
        optimizer.zero_grad()
        timer.start()
        model.train()
        train_itr = iter(train_loader)
        for i in range(steps_per_epoch):
            batch = next(train_itr)

            if steps > unfreeze_after_step and hasattr(model, 'freeze') and model.freeze:
                logging.info("Unfreezing encoders at step %d", steps)
                model.freeze = False
            steps += 1

            x, y = batch
            loss = run_step(x, y, model, loss_function, device, distributed)
            loss.backward()
            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % report_on == 0:
                logging.info(avg_loss)
            if (i + 1) % update_on == 0 and local_rank < 1:
                elapsed = timer.elapsed(True)
                logging.info('elapsed time this epoch %d min', elapsed)
                logging.info('elapsed step time %f steps/min', i / elapsed)
                logging.info('LR: %f', optimizer.current_lr)
                save_checkpoint(model, model_base, steps, tick_type='step', save_npz=save_npz)

        # How much time elapsed in minutes
        elapsed = timer.elapsed(True)
        train_avg_loss = avg_loss.avg
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_avg_loss
        if local_rank < 1:
            avg_valid_loss = Average('average_valid_loss')
            timer.start()
            model.eval()
            valid_itr = iter(valid_loader)
            for j in range(valid_steps):
                with torch.no_grad():
                    batch = next(valid_itr)
                    x, y = batch
                    loss = run_step(x, y, model, loss_function, device, distributed)
                avg_valid_loss.update(loss.item())

            valid_avg_loss = avg_valid_loss.avg

            elapsed = timer.elapsed(True)
            metrics['valid_elapsed_min'] = elapsed

            metrics['average_valid_loss'] = valid_avg_loss
            logger.info(metrics)
            save_checkpoint(model, model_base, epoch, tick_type='epoch', save_npz=save_npz)


def reload_from_checkpoint(model_type, restart_from, restart_tick_type, model, steps_per_epoch):
    if os.path.isdir(restart_from):
        restart_from, _ = find_latest_checkpoint(restart_from)
        print(f'Latest checkpoint: {restart_from}')
    vec = restart_from.split("-")
    try:
        step_num = int(vec[-1].split(".")[0])
    except:
        step_num = 0
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
    main()
