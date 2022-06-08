import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor


class AimCLR_3views_Processor(PT_Processor):
    """
        Processor for 3view-AimCLR Pretraining.
    """
    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_motion_value = []
        loss_bone_value = []

        for [data1, data2, data3], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            data3 = data3.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # forward
            if epoch <= self.arg.mine_epoch:
                output, output_motion, output_bone, target, logits_e, labels_e, logits_motion_e, \
                labels_motion_e, logits_bone_e, labels_bone_e = self.model(data1, data2, data3, cross=False, mine=False)

                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))

                loss_j = self.loss(output, target)
                loss_m = self.loss(output_motion, target)
                loss_b = self.loss(output_bone, target)

                loss2_j = -torch.mean(torch.sum(torch.log(logits_e) * labels_e, dim=1))  # DDM loss
                loss2_m = -torch.mean(torch.sum(torch.log(logits_motion_e) * labels_motion_e, dim=1))  # DDM loss
                loss2_b = -torch.mean(torch.sum(torch.log(logits_bone_e) * labels_bone_e, dim=1))  # DDM loss

                loss_joint = loss_j + loss2_j
                loss_motion = loss_m + loss2_m
                loss_bone = loss_b + loss2_b

            elif epoch <= self.arg.cross_epoch:

                logits, pos_mask_j, logits_motion, pos_mask_m, logits_bone, pos_mask_b,\
                logits_e, labels_e, logits_motion_e, \
                labels_motion_e, logits_bone_e, labels_bone_e = self.model(data1, data2, data3, cross=False, mine=True,
                                                                           topk=self.arg.topk1, vote=self.arg.vote)

                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(logits.size(0))
                else:
                    self.model.update_ptr(logits.size(0))

                loss_j = - (F.log_softmax(logits, dim=1) * pos_mask_j).sum(1) / pos_mask_j.sum(1)
                loss_m = - (F.log_softmax(logits_motion, dim=1) * pos_mask_m).sum(1) / pos_mask_m.sum(1)
                loss_b = - (F.log_softmax(logits_bone, dim=1) * pos_mask_b).sum(1) / pos_mask_b.sum(1)
                
                loss_j = loss_j.mean()
                loss_m = loss_m.mean()
                loss_b = loss_b.mean()

                loss2_j = -torch.mean(torch.sum(torch.log(logits_e) * labels_e, dim=1))  # DDM loss
                loss2_m = -torch.mean(torch.sum(torch.log(logits_motion_e) * labels_motion_e, dim=1))  # DDM loss
                loss2_b = -torch.mean(torch.sum(torch.log(logits_bone_e) * labels_bone_e, dim=1))  # DDM loss

                loss_joint = loss_j + loss2_j
                loss_motion = loss_m + loss2_m
                loss_bone = loss_b + loss2_b
            else:
                logits, logits_motion, logits_bone, pos_mask, \
                logits_e, labels_e, logits_motion_e, \
                labels_motion_e, logits_bone_e, labels_bone_e = self.model(data1, data2, data3, cross=True, mine=False, topk=self.arg.topk2, vote=self.arg.vote)

                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(logits.size(0))
                else:
                    self.model.update_ptr(logits.size(0))

                loss_j = - (F.log_softmax(logits, dim=1) * pos_mask).sum(1) / pos_mask.sum(1)
                loss_m = - (F.log_softmax(logits_motion, dim=1) * pos_mask).sum(1) / pos_mask.sum(1)
                loss_b = - (F.log_softmax(logits_bone, dim=1) * pos_mask).sum(1) / pos_mask.sum(1)

                loss_j = loss_j.mean()
                loss_m = loss_m.mean()
                loss_b = loss_b.mean()

                loss2_j = -torch.mean(torch.sum(torch.log(logits_e) * labels_e, dim=1))  # DDM loss
                loss2_m = -torch.mean(torch.sum(torch.log(logits_motion_e) * labels_motion_e, dim=1))  # DDM loss
                loss2_b = -torch.mean(torch.sum(torch.log(logits_bone_e) * labels_bone_e, dim=1))  # DDM loss

                loss_joint = loss_j + loss2_j
                loss_motion = loss_m + loss2_m
                loss_bone = loss_b + loss2_b

            loss = loss_joint + loss_motion + loss_bone
            self.iter_info['loss'] = loss_joint.data.item()
            self.iter_info['loss_motion'] = loss_motion.data.item()
            self.iter_info['loss_bone'] = loss_bone.data.item()

            loss_value.append(self.iter_info['loss'])
            loss_motion_value.append(self.iter_info['loss_motion'])
            loss_bone_value.append(self.iter_info['loss_bone'])

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)
            self.train_writer.add_scalar('batch_loss_motion', self.iter_info['loss_motion'], self.global_step)
            self.train_writer.add_scalar('batch_loss_bone', self.iter_info['loss_bone'], self.global_step)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.epoch_info['train_mean_loss_motion']= np.mean(loss_motion_value)
        self.epoch_info['train_mean_loss_bone']= np.mean(loss_bone_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss_motion', self.epoch_info['train_mean_loss_motion'], epoch)
        self.train_writer.add_scalar('loss_bone', self.epoch_info['train_mean_loss_bone'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--mine_epoch', type=int, default=1e6, help='the starting epoch of mining top-k')
        parser.add_argument('--cross_epoch', type=int, default=1e6, help='the starting epoch of cross-stream training')
        parser.add_argument('--topk1', type=int, default=1, help='topk samples in cross-stream training')
        parser.add_argument('--topk2', type=int, default=1, help='topk samples in cross-stream training')
        parser.add_argument('--vote', type=int, default=2, help='vote in cross-stream training')
        # endregion yapf: enable

        return parser
