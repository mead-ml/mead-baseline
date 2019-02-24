import six
import argparse
from copy import deepcopy
from itertools import chain
from baseline.utils import read_config_stream, normalize_backend, str2bool
import mead
from mead.utils import convert_path, parse_extra_args
from baseline.train import register_training_func, create_trainer

import matplotlib.pyplot as plt
import numpy as np


@register_training_func('classify', name='lr-find')
@register_training_func('tagger', name='lr-find')
def fit(model, ts, vs, es, **kwargs):
    num_iters = kwargs.get('num_iters', 5)
    kwargs['warmup_steps'] = num_iters * len(ts)
    kwargs['lr'] = kwargs.get('max_lr', 10.)
    trainer = create_trainer(model, **kwargs)
    lrs = []
    losses = []
    use_val = kwargs.get('use_val', False)
    beta = kwargs.get('smooth_beta', 0.05)
    log = kwargs.get('log_scale', True)
    best_loss = six.MAXSIZE
    diverge_threshold = kwargs.get('diverge_threshold', 5)
    stop = False
    i = 0
    be = kwargs.get('backend', 'tf')
    if be == 'tf':
        import tensorflow as tf
        tables = tf.tables_initializer()
        model.sess.run(tables)
        model.sess.run(tf.global_variables_initializer())
        model.set_saver(tf.train.Saver())


    for _ in range(num_iters):
        if stop:
            break
        for batch in ts:
            i += 1
            train_metrics = trainer.train([batch], [])
            if use_val:
                val_metrics = trainer.test(vs, [])
                loss = val_metrics['avg_loss']
            else:
                loss = train_metrics['avg_loss']

            if losses and beta > 0:
                loss = beta * loss + (1 - beta) * losses[-1]
                loss /= (1 - beta ** i)

            losses.append(loss)
            if be == 'tf':
                lrs.append(model.sess.run("OptimizeLoss/lr:0"))
            else:
                lrs.append(trainer.optimizer.current_lr)

            if loss < best_loss:
                best_loss = loss

            if loss > diverge_threshold * best_loss:
                print("Stopping Early")
                stop = True
                break

    plt.plot(lrs, losses)
    if log:
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
    else:
        plt.xlabel('Learning Rate')
    if use_val:
        plt.ylabel('Average Validation Loss')
    else:
        plt.ylabel('Per Batch Training Loss')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train a text classifier')
    parser.add_argument('--config', help='JSON Configuration for an experiment', type=convert_path, default="$MEAD_CONFIG")
    parser.add_argument('--settings', help='JSON Configuration for mead', default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
    parser.add_argument('--embeddings', help='json library of embeddings', default='config/embeddings.json', type=convert_path)
    parser.add_argument('--logging', help='json file for logging', default='config/logging.json', type=convert_path)
    parser.add_argument('--task', help='task to run', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    parser.add_argument('--backend', help='The deep learning backend to use')

    parser.add_argument('--num_iters', type=int, default=5)
    parser.add_argument('--max_lr', type=float, default=10)
    parser.add_argument('--smooth', type=float, default=0.05)
    parser.add_argument('--use_val', type=str2bool, default=False)
    parser.add_argument('--log', type=str2bool, default=True)
    parser.add_argument('--diverge_threshold', type=int, default=5)

    args, reporting_args = parser.parse_known_args()

    config_params = read_config_stream(args.config)
    try:
        args.settings = read_config_stream(args.settings)
    except:
        print('Warning: no mead-settings file was found at [{}]'.format(args.config))
        args.settings = {}
    args.datasets = read_config_stream(args.datasets)
    args.embeddings = read_config_stream(args.embeddings)
    args.logging = read_config_stream(args.logging)

    if args.backend is not None:
        config_params['backend'] = normalize_backend(args.backend)

    config_params['reporting'] = {}
    config_params['train']['fit_func'] = "lr-find"
    config_params['train']['lr_scheduler_type'] = 'warmup_linear'
    config_params['train']['smooth_beta'] = args.smooth
    config_params['train']['use_val'] = args.use_val
    config_params['train']['log_scale'] = args.log
    config_params['train']['diverge_threshold'] = args.diverge_threshold
    config_params['train']['be'] = config_params['backend']

    task_name = config_params.get('task', 'classify') if args.task is None else args.task
    print('Task: [{}]'.format(task_name))
    task = mead.Task.get_task_specific(task_name, args.logging, args.settings)
    task.read_config(config_params, args.datasets, reporting_args=reporting_args, config_file=deepcopy(config_params))
    task.initialize(args.embeddings)
    task.train()


if __name__ == "__main__":
    main()
