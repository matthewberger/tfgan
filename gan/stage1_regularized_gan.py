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

from data.tfdataset import TFDataset
from data.randomparameters import RandomParameters
from model.rendernet import OpacityNetGeneratorAE, OpacityNetDiscriminator

import time
import gansetup

file_dir = os.path.dirname(os.path.realpath(__file__))
data_generator_dir = os.path.abspath(os.path.join(file_dir, os.pardir)) + '/data_generator'
sys.path.insert(0, data_generator_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--min_spatial_res', type=int, default=2, help='minimum resolution to encode to / decode from')
parser.add_argument('--max_channels', type=int, default=512, help='maximum number of channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nof', type=int, default=512)
parser.add_argument('--nvf', type=int, default=512)
parser.add_argument('--latent_dim', type=int, default=8)
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lam', type=float, default=0.01, help='regularization term, default=0.01')
parser.add_argument('--gid', type=int, default=0, help='gpu device id')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gNet', default='', help="path to gNet (to continue training)")
parser.add_argument('--dNet', default='', help="path to dNet (to continue training)")
parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')
parser.add_argument('--name', default='op', help='name of thing for the thing')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--checkpoint_dir', default="/mnt/storage1/berger/th_checkpoints/")

# booleans and their defaults
parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
parser.add_argument('--regularization', dest='regularization', action='store_true', help='enables regularization')
parser.add_argument('--no-regularization', dest='regularization', action='store_false', help='disables regularization')
parser.add_argument('--mismatch', dest='mismatch', action='store_true', help='enable mismatch term in loss')
parser.add_argument('--no-mismatch', dest='mismatch', action='store_false', help='disable mismatch term in loss')
parser.add_argument('--latent', dest='latent', action='store_true', help='enable latent sampling in training')
parser.add_argument('--no-latent', dest='latent', action='store_false', help='disable latent sampling in training')
parser.add_argument('--num_samples', type=int,default=-1, help="size of training data")
parser.set_defaults(cuda=True)
parser.set_defaults(regularization=False)
parser.set_defaults(mismatch=False)
parser.set_defaults(latent=False)

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
    torch.cuda.set_device(opt.gid)
    cudnn.benchmark = True

# data
dataset = TFDataset(opt, train_size=-1, n_channels=1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.num_workers))
fake_weight = 0.5 if opt.mismatch else 1.0

# net setup
gNet, dNet, checkpoint_threshold, lr_decay_threshold, n_samples_processed, n_checkpoints_recorded, n_checkpoint, n_lr_decay = gansetup.setup_training(opt,dataset,OpacityNetGeneratorAE,OpacityNetDiscriminator,5)
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

criterion = nn.BCELoss()
mse_criterion = nn.MSELoss()

# minibatch data
input_view = torch.FloatTensor(opt.batchSize, 5)
input_opacity_tf = torch.FloatTensor(opt.batchSize, 1, 256)
input_img = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)

if opt.cuda:
    criterion.cuda()
    mse_criterion.cuda()
    input_view = input_view.cuda()
    input_opacity_tf = input_opacity_tf.cuda()
    input_img = input_img.cuda()
#

if opt.cuda:
    sampled_view = Variable(torch.FloatTensor(opt.batchSize, 5).cuda())
    sampled_opacity_tf = Variable(torch.FloatTensor(opt.batchSize, 1, 256).cuda())
    sampled_opacity_img = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).cuda())
else:
    sampled_view = Variable(torch.FloatTensor(opt.batchSize, 5))
    sampled_opacity_tf = Variable(torch.FloatTensor(opt.batchSize, 1, 256))
    sampled_opacity_img = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize))

input_view = Variable(input_view)
input_opacity_tf = Variable(input_opacity_tf)
input_img = Variable(input_img)

one = torch.FloatTensor([1]).cuda()
mone = one*-1

real_label = Variable(torch.FloatTensor(opt.batchSize).cuda())
fake_label = Variable(torch.FloatTensor(opt.batchSize).cuda())
fool_label = Variable(torch.FloatTensor(opt.batchSize).cuda())

optimizerD = optim.Adam(dNet.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(gNet.parameters(), lr=opt.lr, betas=(0.5, 0.999))

num_D_params, num_G_params = 0, 0
for layer in dNet.parameters():
    num_D_params += layer.numel()
for layer in gNet.parameters():
    num_G_params += layer.numel()
print('number of parameters discriminator:', num_D_params, 'generator:', num_G_params)

log_file = open('log-' + opt.name + '.txt', 'w')

use_bce_loss = True

def discriminator_gradient_regularization(view_data,op_tf_data,img_data,is_real):
    batch_size = view_data.size()[0]
    concat_grads = True

    view_var = Variable(view_data, requires_grad=True)
    op_tf_var = Variable(op_tf_data, requires_grad=True)
    img_var = Variable(img_data, requires_grad=True)
    decision,raw_decision = dNet([view_var,op_tf_var,img_var])
    all_ones = torch.ones(raw_decision.size()).cuda()

    gradients = autograd.grad(outputs=raw_decision, inputs=[view_var,op_tf_var,img_var], grad_outputs=all_ones, create_graph=True, retain_graph=True, only_inputs=True)
    flattened_op_grad = gradients[1].view(batch_size,gradients[1].size()[1]*gradients[1].size()[2])
    flattened_img_grad = gradients[2].view(batch_size,gradients[2].size()[1]*gradients[2].size()[2]*gradients[2].size()[3])
    if concat_grads:
        flattened_grads = torch.cat((gradients[0],flattened_op_grad,flattened_img_grad),1)
        grads_sqd_norm = torch.sum((flattened_grads**2),1)
    else:
        print('TODO!')
        '''
        view_grad_norm = gradients[0].norm(2,1)
        op_grad_norm = flattened_op_grad.norm(2,1)
        color_grad_norm = flattened_color_grad.norm(2,1)
        img_grad_norm = flattened_img_grad.norm(2,1)
        #print('average gradient norms: view',view_grad_norm.mean().data,'op:',op_grad_norm.mean().data,'color:',color_grad_norm.mean().data,'img:',img_grad_norm.mean().data)
        gradient_penalty = opt.lam*( ((view_grad_norm-1)**2).mean() + ((op_grad_norm-1)**2).mean() + ((color_grad_norm-1)**2).mean() + ((img_grad_norm-1)**2).mean())
        #gradient_penalty = opt.lam*( (((view_grad_norm-1)**2) + ((op_grad_norm-1)**2) + ((color_grad_norm-1)**2) + ((img_grad_norm-1)**2)).mean() )
        '''
    if is_real:
        reg = ((1-decision[:,0])**2)*grads_sqd_norm
    else:
        reg = ((decision[:,0])**2)*grads_sqd_norm
    reg = opt.lam*torch.mean(reg)

    reg.backward(one)

    return reg.data[0]

def generator_gradient_regularization(view_data,op_tf_data):
    batch_size = view_data.size()[0]

    g_view_var = Variable(view_data, requires_grad=True)
    g_op_tf_var = Variable(op_tf_data, requires_grad=True)
    d_view_var = Variable(view_data, requires_grad=True)
    d_op_tf_var = Variable(op_tf_data, requires_grad=True)
    img_var = gNet([g_view_var,g_op_tf_var])
    decision,raw_decision = dNet([d_view_var,d_op_tf_var,img_var])
    all_ones = torch.ones(raw_decision.size()).cuda()

    gradients = autograd.grad(outputs=raw_decision, inputs=[d_view_var,d_op_tf_var,img_var], grad_outputs=all_ones, create_graph=True, retain_graph=True, only_inputs=True)
    flattened_op_grad = gradients[1].view(batch_size,gradients[1].size()[1]*gradients[1].size()[2])
    flattened_img_grad = gradients[2].view(batch_size,gradients[2].size()[1]*gradients[2].size()[2]*gradients[2].size()[3])
    flattened_grads = torch.cat((gradients[0],flattened_op_grad,flattened_img_grad),1)
    grads_sqd_norm = torch.sum((flattened_grads**2),1)

    #for name,layer in gNet.named_parameters():
    #    print('G pre layer',name,'grad norm:',layer.grad.data.norm())

    reg = ((decision[:,0])**2)*grads_sqd_norm
    reg = opt.lam*torch.mean(reg)
    reg.backward(mone)

    #for name,layer in gNet.named_parameters():
    #    print('G post layer',name,'grad norm:',layer.grad.data.norm())

    return reg.data[0]

epoch_start = n_checkpoints_recorded
for epoch in range(epoch_start, opt.niter):
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
        if opt.latent or opt.mismatch:
            mb_view,mb_opacity,mb_img, s_view,s_opacity_tf,s_op_img = data
        else:
            mb_view,mb_opacity,mb_img = data
        batch_size = mb_view.size(0)

        # copy minibatch
        input_view.data.resize_(mb_view.size()).copy_(mb_view)
        input_opacity_tf.data.resize_(mb_opacity.size()).copy_(mb_opacity)
        input_img.data.resize_(mb_img.size()).copy_(mb_img)

        # optionally, copy sampled minibatch
        if opt.latent or opt.mismatch:
            sampled_view.data.resize_(s_view.size()).copy_(s_view)
            sampled_opacity_tf.data.resize_(s_opacity_tf.size()).copy_(s_opacity_tf)
            sampled_opacity_img.data.resize_(s_op_img.size()).copy_(s_op_img)

        # --- discriminator --- #
        #for p in dNet.parameters():
        #    p.requires_grad = True

        # train with real
        dNet.zero_grad()
        real_decision,_ = dNet([input_view,input_opacity_tf,input_img])
        num_real_correct = torch.sum(real_decision.data.cpu()>0.5)

        # real log-loss
        if use_bce_loss:
            real_label.data.resize_(batch_size).fill_(1.0)
            real_log_loss = criterion(real_decision, real_label)
        else:
            real_log_loss = torch.mean(torch.log(real_decision))
            real_log_loss.backward(mone)

        # if regularization enabled, then compute weighted gradient and go backwards
        if opt.regularization and opt.lam > 0:
            time_start = time.time()
            reg_term = discriminator_gradient_regularization(input_view.data,input_opacity_tf.data,input_img.data,True)
            print('real regularization time:',(time.time()-time_start),':',reg_term)
        #

        # train with fake
        fake_img,reconstructed_op_tf = gNet([sampled_view,sampled_opacity_tf]) if opt.latent else gNet([input_view,input_opacity_tf])
        detached_fake_img = fake_img.detach()
        fake_decision,_ = dNet([sampled_view,sampled_opacity_tf,detached_fake_img]) if opt.latent else dNet([input_view,input_opacity_tf,detached_fake_img])
        num_fake_correct = torch.sum(fake_decision.data.cpu()<0.5)

        # fake log-loss
        if use_bce_loss:
            fake_label.data.resize_(batch_size).fill_(0.0)
            fake_log_loss = fake_weight*criterion(fake_decision, fake_label)
        else:
            fake_log_loss = fake_weight*torch.mean(torch.log(1.0 - fake_decision))
            fake_log_loss.backward(mone)

        # backprop for BCE loss
        if use_bce_loss:
            discrim_loss = real_log_loss + fake_log_loss
            discrim_loss.backward()

        # and do regularization
        if opt.regularization and opt.lam > 0:
            time_start = time.time()
            reg_term = discriminator_gradient_regularization(sampled_view.data,sampled_opacity_tf.data,detached_fake_img.data,False)
            print('fake regularization time:',(time.time()-time_start),reg_term)
        #

        # optionally, do mismatch loss
        if opt.mismatch:
            mismatch_decision,_ = dNet([sampled_view,sampled_opacity_tf,input_img])
            num_mismatch_correct = torch.sum(mismatch_decision.data.cpu()<0.5)
            if use_bce_loss:
                mismatch_log_loss = (1.0-fake_weight)*criterion(mismatch_decision, fake_label)
                mismatch_log_loss.backward()
            else:
                mismatch_log_loss = (1.0-fake_weight)*torch.mean(torch.log(1.0 - mismatch_decision))
                mismatch_log_loss.backward(mone)

        optimizerD.step()

        # --- generator --- #
        #for p in dNet.parameters():
        #    p.requires_grad = False
        gNet.zero_grad()
        fool_decision,_ = dNet([sampled_view,sampled_opacity_tf,fake_img]) if opt.latent else dNet([input_view,input_opacity_tf,fake_img])
        num_fool_correct = torch.sum(fool_decision.data.cpu()>0.5)

        # log-loss backwards
        if use_bce_loss:
            fool_label.data.resize_(batch_size).fill_(1.0)
            fool_log_loss = criterion(fool_decision, fool_label)
            ae_loss = mse_criterion(reconstructed_op_tf,input_opacity_tf)
            total_loss = fool_log_loss + ae_loss
            total_loss.backward()
        else:
            fool_log_loss = torch.mean(torch.log(fool_decision))
            fool_log_loss.backward(mone)

        # and do regularization
        if opt.regularization and opt.lam > 0:
            time_start = time.time()
            reg_term = generator_gradient_regularization(sampled_view.data,sampled_opacity_tf.data)
            print('fool regularization time:',(time.time()-time_start),reg_term)
        #

        optimizerG.step()
        if opt.mismatch:
            print('iteration[', epoch, '][', bdx, '] real correct:',num_real_correct, 'fake correct:',num_fake_correct, 'mismatch correct:',num_mismatch_correct, 'fool correct:',num_fool_correct)
        else:
            print('iteration[', epoch, '][', bdx, '] real correct:',num_real_correct, 'fake correct:',num_fake_correct, 'fool correct:',num_fool_correct)
        print('batch time:',(time.time()-batch_start_time))

        # update number of samples processed
        n_samples_processed += batch_size

        if write_images and bdx % 50 == 0:
            fake,_ = gNet([input_view, input_opacity_tf])
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

        D_mean_err = real_log_loss.data[0]+fake_log_loss.data[0]+mismatch_log_loss.data[0] if opt.mismatch else real_log_loss.data[0]+fake_log_loss.data[0]
        G_mean_err = fool_log_loss.data[0]
        if not use_bce_loss:
            D_mean_err = -D_mean_err
            G_mean_err = -G_mean_err
        log_file.write('G_mean_err: ' + str(G_mean_err) + ' D_mean_err: ' + str(D_mean_err) + ' G_grad_norm:' + str(mean_grad_norm(gNet.named_parameters())) + '\n')
        #log_file.write("G_mean_err: " + str(G_mean_err) + " D_mean_err: " + str(D_mean_err) + "\n")
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
