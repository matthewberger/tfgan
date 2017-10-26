from __future__ import print_function
import argparse
import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.autograd as autograd

import time

# custom weights initialization called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#

def setup_generator_training(opt, dataset, Generator, decay_freq=5):
    checkpoint_threshold = len(dataset)
    lr_decay_threshold = decay_freq * checkpoint_threshold

    if opt.gNet is '':
        gNet = Generator(opt)
        gNet.apply(weights_init)
        n_samples_processed, n_checkpoints_recorded, n_checkpoint, n_lr_decay = 0, 0, 0, lr_decay_threshold
    else:
        gNet = torch.load(opt.gNet)
        # ASSUMPTION THAT THE NETS ARE AT THE SAME EPOCH!
        n_checkpoints_recorded = int(opt.gNet.split('-')[-1].split('.')[0])
        n_samples_processed = n_checkpoints_recorded*checkpoint_threshold
        n_checkpoint = (n_checkpoints_recorded+1)*checkpoint_threshold
        n_lr_decay = lr_decay_threshold * (1+int(n_samples_processed/lr_decay_threshold))
        n_checkpoints_recorded+=1

    if opt.cuda:
        gNet.cuda()

    gNet.train()

    cur_decay = 1+int(n_samples_processed/lr_decay_threshold)
    opt.lr /= cur_decay

    return gNet, checkpoint_threshold, lr_decay_threshold, n_samples_processed, n_checkpoints_recorded, n_checkpoint, n_lr_decay

def setup_training(opt, dataset, Generator, Discriminator, decay_freq=5):
    checkpoint_threshold = len(dataset)
    lr_decay_threshold = decay_freq * checkpoint_threshold

    if opt.gNet is '' or opt.dNet is '':
        gNet = Generator(opt)
        gNet.apply(weights_init)
        dNet = Discriminator(opt)
        dNet.apply(weights_init)
        n_samples_processed, n_checkpoints_recorded, n_checkpoint, n_lr_decay = 0, 0, 0, lr_decay_threshold
    else:
        gNet = torch.load(opt.gNet)
        dNet = torch.load(opt.dNet)
        # ASSUMPTION THAT THE NETS ARE AT THE SAME EPOCH!
        n_checkpoints_recorded = int(opt.gNet.split('-')[-1].split('.')[0])
        n_samples_processed = n_checkpoints_recorded*checkpoint_threshold
        n_checkpoint = (n_checkpoints_recorded+1)*checkpoint_threshold
        n_lr_decay = lr_decay_threshold * (1+int(n_samples_processed/lr_decay_threshold))
        n_checkpoints_recorded+=1

    if opt.cuda:
        dNet.cuda()
        gNet.cuda()

    gNet.train()
    dNet.train()

    cur_decay = 1+int(n_samples_processed/lr_decay_threshold)
    opt.lr /= cur_decay

    return gNet, dNet, checkpoint_threshold, lr_decay_threshold, n_samples_processed, n_checkpoints_recorded, n_checkpoint, n_lr_decay
