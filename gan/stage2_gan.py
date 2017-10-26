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

from data.trandataset import TranslationDataset
from model.rendernet import TranslationNetGenerator, TranslationNetDiscriminator

import gansetup

import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--opacityImageSize', type=int, default=64, help='the height / width of the opacity input image to network')
parser.add_argument('--rgbImageSize', type=int, default=256, help='the height / width of the rgb input image to network')
parser.add_argument('--jitter', type=int, default=0, help='crop jitter prior to reshaping for data augmentation')
parser.add_argument('--min_spatial_res', type=int, default=8, help='minimum resolution to encode to / decode from')
parser.add_argument('--max_channels', type=int, default=512, help='maximum number of channels')
parser.add_argument('--nbottleneck', type=int, default=4)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ntf', type=int, default=256)
parser.add_argument('--nvf', type=int, default=256)
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00008, help='learning rate, default=0.00008')
parser.add_argument('--l1', type=float, default=150, help='l1 loss, default=150')
parser.add_argument('--noise_dim', type=int, default=0, help='dimensionality of noise factors, default=0')
parser.add_argument('--gid', type=int, default=0, help='gpu device id')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gNet', default='', help="path to gNet (to continue training)")
parser.add_argument('--dNet', default='', help="path to dNet (to continue training)")
parser.add_argument('--opNet', default='', help="path to opacity network, if specified use it to encode opacity TF for generator")
parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')
parser.add_argument('--name', default='translation', help='name of thing for the thing')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--checkpoint_dir', default="/mnt/storage1/berger/th_checkpoints/")

# booleans and their defaults
parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
parser.add_argument('--rbupsample', dest='rbupsample', action='store_true', help='enables residual block upsampling')
parser.add_argument('--no-rbupsample', dest='rbupsample', action='store_false', help='disables residual block upsampling')
parser.add_argument('--num_samples', type=int, default=-1,help="training data size")
parser.set_defaults(cuda=True)
parser.set_defaults(rbupsample=True)

opt = parser.parse_args()
print(opt)
if opt.checkpoint_dir[-1] != '/':
    opt.checkpoint_dir += '/'
checkpoint_dir = opt.checkpoint_dir + opt.name
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_dir += '/'

write_images = opt.outf is not ''
if write_images:
    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)
    if opt.outf[-1] != '/':
        opt.outf += '/'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    cudnn.benchmark = True
    torch.cuda.set_device(opt.gid)

# data
dataset = TranslationDataset(opt, train_size=-1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.num_workers))
fake_weight = 1.0

# net setup
gNet, dNet, checkpoint_threshold, lr_decay_threshold, n_samples_processed, n_checkpoints_recorded, n_checkpoint, n_lr_decay = gansetup.setup_training(opt,dataset,TranslationNetGenerator,TranslationNetDiscriminator,8)
print('training setup: n_samples_processed:',n_samples_processed,'n_checkpoints_recorded:',n_checkpoints_recorded,'n_checkpoint:',n_checkpoint,'n_lr_decay:',n_lr_decay,'lr:',opt.lr)

def mean_grad_norm(net_params):
    mean_g_norm = 0
    total_params = 0
    for name,layer in net_params:
        total_params += 1
        if layer.grad is not None:
            mean_g_norm += layer.grad.data.norm()
        else:
            print('layer',name,'has no grad?')
    return mean_g_norm

# minibatch data
input_view = torch.FloatTensor(opt.batchSize, 5)
input_opacity_tf = torch.FloatTensor(opt.batchSize, 1, 256)
input_color_tf = torch.FloatTensor(opt.batchSize, 3, 256)
input_opacity_img = torch.FloatTensor(opt.batchSize, 1, opt.opacityImageSize, opt.opacityImageSize)
input_rgb_img = torch.FloatTensor(opt.batchSize, 3, opt.rgbImageSize, opt.rgbImageSize)

use_l1_loss = opt.l1 > 0
l1_lambda = opt.l1

criterion = nn.BCELoss()
l1_criterion = nn.L1Loss()

if opt.cuda:
    criterion.cuda()
    l1_criterion.cuda()

    input_view = input_view.cuda()
    input_opacity_tf = input_opacity_tf.cuda()
    input_color_tf = input_color_tf.cuda()
    input_opacity_img = input_opacity_img.cuda()
    input_rgb_img = input_rgb_img.cuda()
#

if opt.cuda:
    sampled_view = Variable(torch.FloatTensor(opt.batchSize, 5).cuda())
    sampled_opacity_tf = Variable(torch.FloatTensor(opt.batchSize, 1, 256).cuda())
    sampled_color_tf = Variable(torch.FloatTensor(opt.batchSize, 3, 256).cuda())
    sampled_opacity_img = Variable(torch.FloatTensor(opt.batchSize, 1, opt.opacityImageSize, opt.opacityImageSize).cuda())
else:
    sampled_view = Variable(torch.FloatTensor(opt.batchSize, 5))
    sampled_opacity_tf = Variable(torch.FloatTensor(opt.batchSize, 1, 256))
    sampled_color_tf = Variable(torch.FloatTensor(opt.batchSize, 3, 256))
    sampled_opacity_img = Variable(torch.FloatTensor(opt.batchSize, 1, opt.opacityImageSize, opt.opacityImageSize))

input_view = Variable(input_view)
input_opacity_tf = Variable(input_opacity_tf)
input_color_tf = Variable(input_color_tf)
input_opacity_img = Variable(input_opacity_img)
input_rgb_img = Variable(input_rgb_img)

real_label = Variable(torch.FloatTensor(opt.batchSize).cuda())
fake_label = Variable(torch.FloatTensor(opt.batchSize).cuda())
fool_label = Variable(torch.FloatTensor(opt.batchSize).cuda())
if opt.noise_dim > 0:
    noise_vector = Variable(torch.FloatTensor(opt.batchSize,opt.noise_dim).cuda())

one = torch.FloatTensor([1]).cuda()
mone = one*-1

optimizerD = optim.Adam(dNet.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(gNet.parameters(), lr=opt.lr, betas=(0.5, 0.999))

num_D_params, num_G_params = 0, 0
for layer in dNet.parameters():
    num_D_params += layer.numel()
for layer in gNet.parameters():
    num_G_params += layer.numel()
print('number of parameters discriminator:', num_D_params, 'generator:', num_G_params)

log_file = open('log-' + opt.name + '.txt', 'w')

epoch_start = n_checkpoints_recorded
for epoch in range(epoch_start,opt.niter):
    for bdx, data in enumerate(dataloader):
        batch_start_time = time.time()
        # halve learning rate every lr_decay_threshold samples processed
        if n_samples_processed >= n_lr_decay:
            n_lr_decay += lr_decay_threshold
            for param_group in optimizerD.param_groups:
                param_group['lr'] /= 2.0
            for param_group in optimizerG.param_groups:
                param_group['lr'] /= 2.0

        # grab minibatch
        mb_view,mb_opacity_tf,mb_color_tf,mb_op_img,mb_rgb_img = data
        batch_size = mb_view.size(0)

        # copy minibatch
        input_view.data.resize_(mb_view.size()).copy_(mb_view)
        input_opacity_tf.data.resize_(mb_opacity_tf.size()).copy_(mb_opacity_tf)
        input_color_tf.data.resize_(mb_color_tf.size()).copy_(mb_color_tf)
        input_opacity_img.data.resize_(mb_op_img.size()).copy_(mb_op_img)
        input_rgb_img.data.resize_(mb_rgb_img.size()).copy_(mb_rgb_img)

        # noise vector?
        if opt.noise_dim > 0:
            noise_vector.data.resize_(batch_size, opt.noise_dim)
            noise_vector.data.normal_(0, 1)

        # train with real
        dNet.zero_grad()
        real_decision,_ = dNet([input_view,input_opacity_tf,input_color_tf,input_rgb_img])
        num_real_correct = torch.sum(real_decision.data.cpu()>0.5)

        # real log-loss
        real_label.data.resize_(batch_size).fill_(1.0)
        real_log_loss = criterion(real_decision, real_label)

        # train with fake
        if opt.noise_dim > 0:
            fake_img = gNet([input_view,input_opacity_tf,input_color_tf,input_opacity_img,noise_vector])
        else:
            fake_img = gNet([input_view,input_opacity_tf,input_color_tf,input_opacity_img])
        detached_fake_img = fake_img.detach()
        fake_decision,_ = dNet([input_view,input_opacity_tf,input_color_tf,detached_fake_img])
        num_fake_correct = torch.sum(fake_decision.data.cpu()<0.5)

        # fake log-loss
        fake_label.data.resize_(batch_size).fill_(0.0)
        fake_log_loss = fake_weight*criterion(fake_decision, fake_label)

        discrim_loss = real_log_loss+fake_log_loss
        discrim_loss.backward()

        optimizerD.step()

        # --- generator --- #
        #for p in dNet.parameters():
        #    p.requires_grad = False
        gNet.zero_grad()
        fool_decision,_ = dNet([input_view,input_opacity_tf,input_color_tf,fake_img])
        num_fool_correct = torch.sum(fool_decision.data.cpu()>0.5)

        # log-loss backwards
        fool_label.data.resize_(batch_size).fill_(1.0)
        fool_log_loss = criterion(fool_decision, fool_label)

        # optionally do l1 loss
        if use_l1_loss:
            l1_loss = l1_lambda*l1_criterion(fake_img,input_rgb_img)
            gen_loss = l1_loss+fool_log_loss
            gen_loss.backward()
        else:
            fool_log_loss.backward()

        optimizerG.step()
        print('iteration[', epoch, '][', bdx, '] real correct:',num_real_correct, 'fake correct:',num_fake_correct, 'fool correct:',num_fool_correct)
        print('batch time:',(time.time()-batch_start_time))

        # update number of samples processed
        n_samples_processed += batch_size

        if write_images and bdx % 50 == 0:
            if opt.noise_dim > 0:
                fake = gNet([input_view, input_opacity_tf, input_color_tf, input_opacity_img, noise_vector])
            else:
                fake = gNet([input_view, input_opacity_tf, input_color_tf, input_opacity_img])
            if opt.cuda:
                fake.data = torch.min(fake.data, torch.ones(fake.data.size()[0], fake.data.size()[
                    1], fake.data.size()[2], fake.data.size()[3]).cuda())
                fake.data = torch.max(fake.data, -1 * torch.ones(fake.data.size()
                                                                 [0], fake.data.size()[1], fake.data.size()[2],
                                                                 fake.data.size()[3]).cuda())
            else:
                fake.data = torch.min(fake.data, torch.ones(fake.data.size()[0], fake.data.size()[
                    1], fake.data.size()[2], fake.data.size()[3]))
                fake.data = torch.max(fake.data, -1 * torch.ones(fake.data.size()
                                                                 [0], fake.data.size()[1], fake.data.size()[2],
                                                                 fake.data.size()[3]))
            batch_img_filename = opt.outf + opt.name + '-fake-' + str(epoch) + '-' + str(bdx) + '.png'
            fake.data = 0.5*(fake.data+torch.ones(fake.data.size()[0],fake.data.size()[1],fake.data.size()[2],fake.data.size()[3]).cuda())
            vutils.save_image(fake.data, batch_img_filename)

        D_mean_err = real_log_loss.data[0]+fake_log_loss.data[0]
        G_mean_err = fool_log_loss.data[0]+l1_loss.data[0] if use_l1_loss else fool_log_loss.data[0]
        if use_l1_loss:
            log_file.write('G_err: ' + str(fool_log_loss.data[0]) + ' lambda*L1_err: ' + str(l1_loss.data[0]) + ' L1_err: ' + str(l1_loss.data[0]/l1_lambda) + ' D_mean_err: ' + str(D_mean_err) + ' G_grad_norm:' + str(mean_grad_norm(gNet.named_parameters())) + '\n')
        else:
            log_file.write('G_mean_err: ' + str(G_mean_err) + ' D_mean_err: ' + str(D_mean_err) + ' G_grad_norm:' + str(mean_grad_norm(gNet.named_parameters())) + '\n')
        log_file.flush()

        # checkpoint models every checkpoint_threshold samples processed
        if n_samples_processed >= n_checkpoint:
            n_checkpoint += checkpoint_threshold
            gNet_checkpoint_filename = checkpoint_dir + opt.name + '-gNet-' + str(n_checkpoints_recorded) + '.pth'
            dNet_checkpoint_filename = checkpoint_dir + opt.name + '-dNet-' + str(n_checkpoints_recorded) + '.pth'
            n_checkpoints_recorded += 1
            with open(gNet_checkpoint_filename, 'wb') as gNet_checkpoint_file:
                torch.save(gNet, gNet_checkpoint_file)
            with open(dNet_checkpoint_filename, 'wb') as dNet_checkpoint_file:
                torch.save(dNet, dNet_checkpoint_file)
#
