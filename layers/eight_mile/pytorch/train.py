"""Training loop utilities

This provides an interface that makes it easy to wire up hooks and train over multiple devices.
To use these utilities, define a `TrainingTarget` with a stepwise definition for training and
validation.


"""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import os
from eight_mile.utils import listify, get_num_gpus_multiworker
from typing import Union, List
from eight_mile.train import MetricObserver, GlobalMetrics
from eight_mile.pytorch.optz import OptimizerManager
from eight_mile.pytorch.layers import save_checkpoint, checkpoint_for, init_distributed
from eight_mile.progress import create_progress_bar

from contextlib import ExitStack
from baseline.utils import get_metric_cmp
from baseline.embeddings import *
import logging

logger = logging.getLogger(__file__)


class TrainingTarget(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = 'cpu'

    def forward(self, batch):
        if self.training:
            return self.train_step(batch)
        return self.eval_step(batch)

    def train_step(self, batch):
        pass

    def eval_step(self, batch):
        pass

    @property
    def model(self):
        pass

    def set_device(self, device):
        self.to(device=device)
        self.device = device


class SaveCheckpoint(MetricObserver):
    def __init__(self, checkpoint_dir, model_base='checkpoint'):
        self.checkpoint_dir = checkpoint_dir
        self.model_base = os.path.join(self.checkpoint_dir, model_base)

    def run(self, model, metrics, global_step):
        save_checkpoint(model, self.model_base, global_step, tick_type='step', rm_wrapper=False)

    def get_model_file(self, global_step):
        return checkpoint_for(self.model_base, global_step, tick_type='step') + '.pth'


class CheckpointManager(MetricObserver):
    # This doesnt actually have to save if we can guarantee there is a save metric in there somewhere
    def __init__(self, checkpoint_dir, model_base='checkpoint', early_stopping_key=None):
        self.early_stopping_key = early_stopping_key
        self.saver = SaveCheckpoint(checkpoint_dir, model_base)
        if self.early_stopping_key:
            self.early_stopping_cmp, self.best_metric = get_metric_cmp(self.early_stopping_key)
        self.step = -1

    def run(self, model, metrics, global_step):
        self.saver.run(model, metrics, global_step)
        if self.early_stopping_key:
            current = metrics[self.early_stopping_key]
            if self.early_stopping_cmp(current, self.best_metric):
                self.step = global_step
                self.best_metric = current
                logger.info('New best %.3f', self.best_metric)
        else:
            self.step = global_step

    def get_model_file(self, global_step=-1):
        if global_step < 1:
            global_step = self.step
        return self.saver.get_model_file(global_step)

    def restore(self, module, global_step=-1, str_map={}, map_location=None):
        checkpoint = self.get_model_file(global_step)
        logger.info('Restoring %s', checkpoint)

        ckpt_dict = torch.load(checkpoint, map_location=map_location)
        renamed = {}
        for k, v in ckpt_dict.items():
            for from_str, to_str in str_map.items():
                k = k.replace(from_str, to_str)
            renamed[k] = v
        unmatch = module.load_state_dict(renamed, strict=False)
        if unmatch.missing_keys or len(unmatch.unexpected_keys) > 2:
            print("Warning: Embedding doesn't match with the checkpoint being loaded.")
            print(f"missing keys: {unmatch.missing_keys}\n unexpected keys: {unmatch.unexpected_keys}")


class Trainer:

    def __init__(
            self,
            train_module,
            global_step: int = 0,
            optim: str='adam',
            lr: float=0.001,
            weight_decay: float = 0.0,
            loss_key: str = 'loss',
            clip: float = 50.0,
            train_metric_observers: Union[List[MetricObserver], MetricObserver] = [],
            valid_metric_observers: Union[List[MetricObserver], MetricObserver] = [],
            test_metric_observers: Union[List[MetricObserver], MetricObserver] = [],
            **kwargs):
        self.train_module = train_module
        self.clip = clip

        self.optimizer = OptimizerManager(train_module, global_step, optim=optim, lr=lr, weight_decay=weight_decay)
        self.loss_key = loss_key
        self.train_metric_observers = listify(train_metric_observers)
        self.valid_metric_observers = listify(valid_metric_observers)
        self.test_metric_observers = listify(test_metric_observers)

    def _fire_train_observers(self, metrics):
        for observer in self.train_metric_observers:
            observer.run(self.train_module, metrics, self.optimizer.global_step)

    def _fire_valid_observers(self, metrics):
        for observer in self.valid_metric_observers:
            observer.run(self.train_module, metrics, self.optimizer.global_step)

    def _fire_test_observers(self, metrics):
        for observer in self.test_metric_observers:
            observer.run(self.train_module, metrics, self.optimizer.global_step)

    def run(self, train_loader, valid_loader=None, eval_loader=None, num_epochs: int = 1,
            report_on: int = 100, grad_accum: int = 1,
            early_stopping_metric: str=None,
            local_rank=0,
            distributed=False,
            basedir: str=None,
            device='cuda',
            max_steps_per_epoch=None,
            progress_bar='default'):

        # Get the basedir to save results and checkpoints
        if basedir is None:
            basedir = f'checkpoints-{os.getpid()}'
        os.makedirs(basedir, exist_ok=True)

        # Setup logger
        logging.basicConfig(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
        num_gpus = get_num_gpus_multiworker()

        distributed = distributed or num_gpus > 1
        if distributed:
            device, updated_local_rank = init_distributed(local_rank)
            local_rank = updated_local_rank

        logger.info(f"Using {num_gpus} GPUs in this job.")

        self.train_module.set_device(device)

        checkpoint_manager = CheckpointManager(basedir, early_stopping_key=early_stopping_metric)
        self.valid_metric_observers.append(checkpoint_manager)

        steps_train = len(train_loader)
        steps_valid = len(valid_loader) if valid_loader else 0
        steps_eval = len(eval_loader) if eval_loader else 0

        if distributed:
            train_module = DistributedDataParallel(self.train_module, device_ids=[device], output_device=device)
        else:
            train_module = self.train_module

        for epoch in range(num_epochs):
            train_module.train()
            steps = steps_train
            if max_steps_per_epoch and max_steps_per_epoch < steps_train:
                steps = max_steps_per_epoch
            pg = create_progress_bar(steps, name=progress_bar)
            epoch_train_metrics = GlobalMetrics()
            last_report_step = -1

            for iters, batch in enumerate(pg(train_loader)):
                is_dist_step = (iters + 1) % grad_accum == 0
                with train_module.no_sync() if (distributed and not is_dist_step) else ExitStack():
                    metrics = train_module(batch)
                    loss = metrics[self.loss_key]
                    epoch_train_metrics.update(metrics)
                    loss.backward()

                    if is_dist_step:
                        if self.clip and self.clip > 0.0:
                            self.optimizer.clip_grads(self.clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if max_steps_per_epoch is not None and (iters / grad_accum) >= (max_steps_per_epoch-1):
                            break

                        # Only fire reporting from the master node for now
                        if self.optimizer.global_step % report_on == 0 and local_rank < 1:
                            last_report_step = self.optimizer.global_step
                            self._fire_train_observers(epoch_train_metrics.reduce())

            # If its a worker, continue, if its master, we still are going
            if steps_valid < 1 or local_rank > 0:
                continue
            if self.optimizer.global_step != last_report_step:
                self._fire_train_observers(epoch_train_metrics.reduce())

            train_module.eval()
            pg = create_progress_bar(steps_valid)
            epoch_valid_metrics = GlobalMetrics()
            for batch in pg(valid_loader):
                metrics = train_module(batch)
                epoch_valid_metrics.update(metrics)

            self._fire_valid_observers(epoch_valid_metrics.reduce())

        # We only are going to evaluate on master
        if steps_eval < 1 or local_rank > 0:
            return

        pg = create_progress_bar(steps_eval)
        epoch_eval_metrics = GlobalMetrics()

        checkpoint_manager.restore(self.train_module, map_location=device)
        train_module.eval()
        for batch in pg(eval_loader):
            metrics = self.train_module(batch)
            epoch_eval_metrics.update(metrics)

        self._fire_test_observers(epoch_eval_metrics.reduce())
