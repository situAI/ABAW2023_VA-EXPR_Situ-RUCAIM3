import torch
from core.data import build_dataloader
from core.model import build_model
from core.optimizer import build_optimizer, build_lr_scheduler
from core.loss import build_loss
from core.metric import build_metric
from utils.registery import SOLVER_REGISTRY
from utils.helper import format_print_dict
from utils.logger import get_logger_and_log_path
import os
import copy
import datetime
from torch.nn.parallel import DistributedDataParallel
import yaml
import numpy as np
import pandas as pd


@SOLVER_REGISTRY.register()
class BaseSolver(object):
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.task = self.cfg['task']
        self.local_rank = torch.distributed.get_rank()
        self.train_loader, self.val_loader = build_dataloader(cfg)
        self.len_train_loader, self.len_val_loader = len(self.train_loader), len(self.val_loader)
        self.criterion = build_loss(cfg).cuda(self.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(build_model(cfg))
        self.model = DistributedDataParallel(model.cuda(self.local_rank), device_ids=[
                                             self.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        self.optimizer = build_optimizer(cfg)(self.model.parameters(), **cfg['solver']['optimizer']['args'])
        self.hyper_params = cfg['solver']['args']
        crt_date = datetime.date.today().strftime('%Y-%m-%d')
        self.logger, self.log_path = get_logger_and_log_path(crt_date=crt_date, **cfg['solver']['logger'])
        self.metric_fn = build_metric(cfg)
        try:
            self.epoch = self.hyper_params['epoch']
        except Exception:
            raise 'should contain epoch in {solver.args}'
        if self.local_rank == 0:
            self.save_dict_to_yaml(self.cfg, os.path.join(self.log_path, 'config.yaml'))
            self.logger.info(self.cfg)
        if 'mix' in self.cfg.keys() and self.cfg['mix'] == True:
            self.epoch_prefix = True
        else:
            self.epoch_prefix = False

    def train(self):
        if torch.distributed.get_rank() == 0:
            self.logger.info('==> Start Training')
        lr_scheduler = build_lr_scheduler(self.cfg)(self.optimizer, **self.cfg['solver']['lr_scheduler']['args'])

        val_v_list = [-1]
        val_a_list = [-1]
        val_peek_list = [-1]
        max_v = -1
        max_a = -1
        max_f1 = -1
        max_v_epoch = -1
        max_a_epoch = -1
        max_f1_epoch = -1

        for t in range(self.epoch):
            self.train_loader.sampler.set_epoch(t)
            if torch.distributed.get_rank() == 0:
                self.logger.info(f'==> epoch {t + 1}')
            self.model.train()

            pred_list = list()
            label_list = list()

            mean_loss = 0.0

            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                feat = data['feat'].cuda(self.local_rank)
                label = data['label'].cuda(self.local_rank)
                seq_list = data['seq_list']

                pred = self.model(feat)

                seq_len, bs, _ = pred.shape
                seq_list = seq_list.reshape((seq_len * bs))
                pred = pred.reshape((seq_len * bs, -1))
                if self.task == 'expr':
                    label = label.reshape((seq_len * bs))
                else:
                    label = label.reshape((seq_len * bs, -1))

                loss = self.criterion(pred, label)
                mean_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                if self.task == 'expr':
                    pred = pred.argmax(dim=-1)

                if (i == 0 or i % 100 == 0) and (torch.distributed.get_rank() == 0):
                    self.logger.info(f'epoch: {t + 1}/{self.epoch}, iteration: {i + 1}/{self.len_train_loader}, loss: {loss.item() :.4f}')

                if i == self.len_train_loader // 2:
                    if torch.distributed.get_rank() == 0:
                        if self.task == 'va':
                            peek_v, peek_a = self.val(t + 0.5)
                            if peek_v >= max(val_v_list):
                                self.save_checkpoint(self.model, self.cfg, self.log_path, t + 0.5, 'valence', self.epoch_prefix)
                                val_v_list.append(peek_v)
                            else:
                                if self.epoch_prefix:
                                    self.save_checkpoint(self.model, self.cfg, self.log_path, t + 0.5, 'valence', self.epoch_prefix)
                                val_v_list.append(peek_v)

                            if peek_a >= max(val_a_list):
                                self.save_checkpoint(self.model, self.cfg, self.log_path, t + 0.5, 'arousal', self.epoch_prefix)
                                val_a_list.append(peek_a)
                            else:
                                if self.epoch_prefix:
                                    self.save_checkpoint(self.model, self.cfg, self.log_path, t + 0.5, 'arousal', self.epoch_prefix)
                                val_a_list.append(peek_a)
                        else:
                            peek = self.val(t + 0.5)
                            if peek > max(val_peek_list):
                                self.save_checkpoint(self.model, self.cfg, self.log_path, t + 0.5, self.task, self.epoch_prefix)
                                val_peek_list.append(peek)
                            else:
                                if self.epoch_prefix:
                                    self.save_checkpoint(self.model, self.cfg, self.log_path, t + 0.5, self.task, self.epoch_prefix)
                                val_peek_list.append(peek)
                    self.model.train()

                batch_pred = [torch.zeros_like(pred) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_pred, pred)
                pred_list.append(torch.cat(batch_pred, dim=0).detach().cpu())

                batch_label = [torch.zeros_like(label) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(batch_label, label)
                label_list.append(torch.cat(batch_label, dim=0).detach().cpu())

            pred_list = torch.cat(pred_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
            pred_list = pred_list.numpy()
            label_list = label_list.numpy()

            metric_dict = self.metric_fn(**{'pred': pred_list, 'gt': label_list})
            mean_loss = mean_loss / self.len_train_loader

            print_dict = dict()
            print_dict.update({'epoch': f'{t + 1}/{self.epoch}'})
            print_dict.update({'mean_loss': mean_loss})
            print_dict.update({'lr': f"{self.optimizer.state_dict()['param_groups'][0]['lr'] :.7f}"})
            print_dict.update(metric_dict)

            print_str = format_print_dict(print_dict)

            if torch.distributed.get_rank() == 0:
                self.logger.info(f"==> train: {print_str}")
                if self.task == 'va':
                    peek_v, peek_a = self.val(t + 1)
                    if peek_v >= max(val_v_list):
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, 'valence', self.epoch_prefix)
                        val_v_list.append(peek_v)
                    else:
                        if self.epoch_prefix:
                            self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, 'valence', self.epoch_prefix)
                        val_v_list.append(peek_v)
                    if peek_a >= max(val_a_list):
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, 'arousal', self.epoch_prefix)
                        val_a_list.append(peek_a)
                    else:
                        if self.epoch_prefix:
                            self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, 'arousal', self.epoch_prefix)
                        val_a_list.append(peek_a)
                else:
                    peek = self.val(t + 1)
                    if peek >= max(val_peek_list):
                        print('if here: t + 1 = {t + 1}')
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, self.task, self.epoch_prefix)
                        val_peek_list.append(peek)
                    else:
                        if self.epoch_prefix:
                            print('else here: t + 1 = {t + 1}')
                            self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, self.task, self.epoch_prefix)
                        val_peek_list.append(peek)

            lr_scheduler.step(t)

        if self.local_rank == 0:
            if self.task == 'va':
                max_v = max(val_v_list)
                max_a = max(val_a_list)
                max_v_epoch = val_v_list.index(max(val_v_list)) / 2
                max_a_epoch = val_a_list.index(max(val_a_list)) / 2
                self.logger.info(
                    f'==> End Training, BEST Valence: {max_v}, BEST Valence Epoch: {max_v_epoch :.1f}, BEST Arousal: {max_a}, BEST Arousal Epoch: {max_a_epoch :.1f}'
                )

            else:
                max_f1 = max(val_peek_list)
                max_f1_epoch = val_peek_list.index(max(val_peek_list)) / 2
                self.logger.info(f'==> End Training, BEST F1: {max_f1}, BEST F1 Epoch: {max_f1_epoch :.1f}')

        if self.task == 'va':
            return max_v, max_v_epoch, max_a, max_a_epoch
        else:
            return max_f1, max_f1_epoch

    @torch.no_grad()
    def val(self, t):
        self.model.eval()

        pred_list = list()
        logits_list = list()
        label_list = list()
        overall_seq_list = list()
        save_dict = dict()
        save_dict['pred_list'] = list()
        save_dict['label_list'] = list()
        save_dict['seq_list'] = list()

        for i, data in enumerate(self.val_loader):
            feat = data['feat'].cuda(self.local_rank)
            label = data['label'].cuda(self.local_rank)
            seq_list = data['seq_list']

            pred = self.model(feat)

            if self.task == 'expr':
                logits = pred.clone()
                logits_list.append(logits.detach().cpu())
                pred = pred.argmax(dim=-1)

            pred_list.append(pred.detach().cpu())
            label_list.append(label.detach().cpu())
            overall_seq_list.append(seq_list)

        pred_list = torch.cat(pred_list, dim=1)
        label_list = torch.cat(label_list, dim=1)
        overall_seq_list = np.concatenate(overall_seq_list, axis=1)
        seq_len, bs = pred_list.shape[:2]
        if self.task == 'expr':
            logits_list = torch.cat(logits_list, dim=1)
            logits_list = logits_list.reshape((seq_len * bs, -1)).numpy()
        pred_list = pred_list.reshape((seq_len * bs, -1))
        label_list = label_list.reshape((seq_len * bs, -1))
        overall_seq_list = overall_seq_list.reshape((seq_len * bs, -1))
        pred_list = pred_list.numpy()
        label_list = label_list.numpy()

        if self.task == 'expr':
            df = dict()
            df['seq_name'] = overall_seq_list.squeeze()
            df['logits'] = logits_list
            df['pred'] = pred_list
            df['label'] = label_list
            df = pd.DataFrame.from_dict(df, orient='index').T
            df = df.drop_duplicates(subset=['seq_name'], keep='first')
            df.sort_values(by=['seq_name'], inplace=True)
        else:
            df = dict()
            df['seq_name'] = overall_seq_list.squeeze()
            df['pred'] = pred_list
            df['label'] = label_list
            df = pd.DataFrame.from_dict(df, orient='index').T
            df = df.drop_duplicates(subset=['seq_name'], keep='first')
            df.sort_values(by=['seq_name'], inplace=True)

        metric_dict = self.metric_fn(**{'pred': np.asarray(df['pred'].tolist()), 'gt': np.asarray(df['label'].tolist())})
        print_dict = dict()
        print_dict.update({'epoch': f'{t}'})
        print_dict.update(metric_dict)
        print_str = format_print_dict(print_dict)

        if torch.distributed.get_rank() == 0:
            self.logger.info(f"==> val: {print_str}")

        if self.task == 'va':
            peek_v = metric_dict['valence_ccc']
            peek_a = metric_dict['arousal_ccc']
            return peek_v, peek_a
        else:
            peek = metric_dict['F1']
            return peek

    def run(self):
        out = self.train()

        return out

    @staticmethod
    def save_dict_to_yaml(dict_value, save_path):
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(dict_value, file, sort_keys=False)

    def save_checkpoint(self, model, cfg, log_path, epoch_id, task_name, epoch_prefix=False):
        model.eval()
        if epoch_prefix:
            torch.save(model.module.state_dict(), os.path.join(log_path, f'ckpt_epoch_{epoch_id}_{task_name}.pt'))
        else:
            torch.save(model.module.state_dict(), os.path.join(log_path, f'ckpt_{task_name}.pt'))
