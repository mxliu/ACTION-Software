#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:44:36 2024

@author: qqw

"""

import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim
import torch.backends.cudnn as cudnn
import math
import time
import shutil
#import model_pretrain
import warnings
warnings.simplefilter('ignore')
import pickle
import numpy as np
import torch
import scipy
import torch.nn as nn
#parser.add_argument(xxx)
#

from einops import repeat, rearrange, reduce
import torch
import numpy as np
####################################Hyper-Parameters####################################
parser = argparse.ArgumentParser(description='PyTorch MDD Training')

parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--dim', default=64, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('--pred-dim', default=32, type=int,
                    help='hidden dimension of the predictor (default: 32)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

################### Here, you can select a specific encoder!!###################
parser.add_argument('--encoder-type',
                    choices=['GCN', 'GIN', 'GAT','BrainNetCNN','Transformer'],
                    default='HGCN',  # Change the default to 'gat'
                    help="Choose the type of encoder")

args = parser.parse_args()

#
Module_1 = getattr(__import__('{}_encoder'.format(args.encoder_type)), 'Module_1')
print(Module_1)



if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

####################################Data Loading####################################
with open('/home/qqw/Downloads/combine_ADHD_ABIDE_MDD_3806subj_TP170_data.pkl', 'rb') as file:
    full_data = pickle.load(file)  # shape  (3806,170,116)



####################################Bold Signal Augmentation####################################
class Data_Aug(object):
    def read_data(self):
        Data_top_90 = []
        Data_bottom_90 = []
        for i in range(full_data.shape[0]):
           # print(i)
            a = full_data[i]
            length = int(full_data.shape[1] * 0.9)
            #print(a.shape)
            data_top_90 = a[:length, :]
            Data_top_90.append(data_top_90)
            data_bottom_90 = a[-length:, :]
            Data_bottom_90.append(data_bottom_90)
        Bold1 = np.array(Data_top_90)
        Bold2 = np.array(Data_bottom_90)
        return Bold1, Bold2

    def __init__(self):
        super(Data_Aug, self).__init__()
        Bold1, Bold2= self.read_data()

        self.Bold1 = torch.from_numpy(Bold1)
        self.Bold2 = torch.from_numpy(Bold2)
        self.n_samples = Bold1.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.Bold1[index], self.Bold2[index]

Dataset=Data_Aug()


###################Contrastive Learning Framework#############

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, dim, pred_dim):
        super(SimSiam, self).__init__()

        self.first_encoder = Module_1(
    input_dim=116,
    hidden_dim=64,
            c=0.0001,  # default from parser
            a_couping=0.01,  # default from parser
            dropout=0.3,  # default from parser
            use_att=False,  # default from parser
            local_agg=True,  # default from parser
            manifold='PoincareBall'  # default from parser
)

        self.predictor = nn.Sequential()
        self.predictor.add_module('L1', nn.Linear(dim, pred_dim, bias=False)),
        self.predictor.add_module('BN', nn.BatchNorm1d(pred_dim)),
        self.predictor.add_module('RL', nn.ReLU(inplace=True)),  # hidden layer
        self.predictor.add_module('L2', nn.Linear(pred_dim, dim))  # output layer

    def forward(self, data1,data2):
    #def forward(self, x1, x2, A1, A2):
        """
        Input:
            data1: first views of signals
            data2: second views of signals
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        z1 = self.first_encoder(data1)  # 16,104,64# NxC
        z2 = self.first_encoder(data2)  # (nbatch,64,1,1) # NxC#
        z1 = reduce(z1, 'b n c ->b c', 'mean')
        z2 = reduce(z2, 'b n c ->b c', 'mean')
        p1 = self.predictor(z1)  # nbatch,64)
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()



def train(train_loader, Model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    Model.train()

    end = time.time()
    for i, (data1,data2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data1 = data1.float().to(device)
        data2 = data2.float().to(device)
        p1, p2, z1, z2 = Model(data1, data2)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        losses.update(loss.item(), data1.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
    print(loss)
    return loss


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = args.lr
        else:
            param_group['lr'] = cur_lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
Model = SimSiam(args.dim, args.pred_dim)#model_pretrain.
Model.to(device)
print(Model)
criterion = nn.CosineSimilarity()

if args.fix_pred_lr:
    optim_params = [{'params': Model.first_encoder.parameters(), 'fix_lr': False},
                    {'params': Model.second_encoder.parameters(), 'fix_lr': False},
                    {'params': Model.predictor.parameters(), 'fix_lr': True}]
else:
    optim_params = Model.parameters()

optimizer = torch.optim.SGD(optim_params, args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(optim_params,args.lr,weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        Model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True
train_loader = torch.utils.data.DataLoader(
    Dataset, batch_size=args.batch_size, shuffle=True)
training_loss = 0.0

losses = []
for epoch in range(args.start_epoch, args.epochs):

    adjust_learning_rate(optimizer, epoch, args)
    loss = train(train_loader, Model, criterion, optimizer, epoch, args)
    print('here', loss)
    losses.append(loss.cpu().detach().numpy())
   # if epoch == 49:
    if (epoch+1) % 10 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': Model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename='Pretrained_{}_{:03d}.pth.tar'.format(args.encoder_type,epoch))


import matplotlib.pyplot as plt
# Plot the loss over epochs
plt.plot(range(1, args.epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('{}.png'.format(args.encoder_type), dpi=300)









