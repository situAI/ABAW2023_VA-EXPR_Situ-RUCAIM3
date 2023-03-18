import torch
from core.optimizer import build_lr_scheduler
from utils.registery import SOLVER_REGISTRY
from utils.helper import format_print_dict
from .base_solver import BaseSolver


@SOLVER_REGISTRY.register()
class RDropSolver(BaseSolver):
    def __init__(self, cfg):
        super().__init__(cfg)

    def train(self):
        if self.task == 'va':
            raise 'RDrop not implement in task: `va` yet!'
        if torch.distributed.get_rank() == 0:
            self.logger.info('==> Start Training')
        lr_scheduler = build_lr_scheduler(self.cfg)(self.optimizer, **self.cfg['solver']['lr_scheduler']['args'])

        val_peek_list = [-1]
        val_v_list = [-1]
        val_a_list = [-1]
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

                logits1 = self.model(feat)
                logits2 = self.model(feat)

                seq_len, bs, _ = logits1.shape
                seq_list = seq_list.reshape((seq_len * bs))
                logits1 = logits1.reshape((seq_len * bs, -1))
                logits2 = logits2.reshape((seq_len * bs, -1))
                if self.task == 'expr':
                    label = label.reshape((seq_len * bs))
                else:
                    label = label.reshape((seq_len * bs, -1))

                loss = self.criterion(logits1, logits2, label)
                mean_loss += loss.item()

                if self.task == 'expr':
                    pred = logits1.argmax(dim=-1)
                else:
                    pred = logits1

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

                loss.backward()
                self.optimizer.step()

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
            print_dict.update({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
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
                        val_v_list.append(peek_v)
                    if peek_a >= max(val_a_list):
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, 'arousal', self.epoch_prefix)
                        val_a_list.append(peek_a)
                    else:
                        val_a_list.append(peek_a)
                else:
                    peek = self.val(t + 1)
                    if peek > max(val_peek_list):
                        self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, self.task, self.epoch_prefix)
                        val_peek_list.append(peek)
                    else:
                        if self.epoch_prefix:
                            self.save_checkpoint(self.model, self.cfg, self.log_path, t + 1, self.task, self.epoch_prefix)
                        val_peek_list.append(peek)

            lr_scheduler.step(t)

        if self.task == 'va':
            max_v = max(val_v_list)
            max_a = max(val_a_list)
            max_v_epoch = val_v_list.index(max(val_v_list)) / 2
            max_a_epoch = val_a_list.index(max(val_a_list)) / 2
            self.logger.info(f'==> End Training, BEST Valence: {max_v}, BEST Valence Epoch: {max_v_epoch :.1f}, BEST Arousal: {max_a}, BEST Arousal Epoch: {max_a_epoch :.1f}')

            return max_v, max_v_epoch, max_a, max_a_epoch
        else:
            max_f1 = max(val_peek_list)
            max_f1_epoch = val_peek_list.index(max(val_peek_list)) / 2
            self.logger.info(f'==> End Training, BEST F1: {max_f1}, BEST F1 Epoch: {max_f1_epoch}')

            return max_f1, max_f1_epoch
