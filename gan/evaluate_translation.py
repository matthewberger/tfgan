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
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.autograd import Variable

import numpy as np
from PIL import Image
from data.trandataset import TranslationDataset
from data.data_manager import DataManager
from model.lr_opacitynet import OpacityNetGenerator

from skimage import io, color
from sklearn.metrics.pairwise import euclidean_distances
import pyemd

from tqdm import tqdm
import time

from ssim import compute_ssim

def compute_emd(im1, im2, cost_mat, l_bins=8,a_bins=12,b_bins=12):
    lab_im1 = color.rgb2lab(im1)
    lab_im1 = lab_im1.reshape((lab_im1.shape[0]*lab_im1.shape[1],lab_im1.shape[2]))
    lab_hist_1,_ = np.histogramdd(lab_im1, bins=(l_bins,a_bins,b_bins), range=[[0.0,100.0], [-86.185,98.254], [-107.863,94.482]], normed=False)

    lab_im2 = color.rgb2lab(im2)
    lab_im2 = lab_im2.reshape((lab_im2.shape[0]*lab_im2.shape[1],lab_im2.shape[2]))
    lab_hist_2,_ = np.histogramdd(lab_im2, bins=(l_bins,a_bins,b_bins), range=[[0.0,100.0], [-86.185,98.254], [-107.863,94.482]], normed=False)

    n_bins = l_bins*a_bins*b_bins
    lab_hist_1 = lab_hist_1.reshape((n_bins))
    lab_hist_2 = lab_hist_2.reshape((n_bins))
    img_res = lab_im1.shape[0]
    lab_hist_1 /= img_res
    lab_hist_2 /= img_res
    return pyemd.emd(lab_hist_1,lab_hist_2,cost_mat)

def compute_emd_cost_mat(l_bins=8,a_bins=12,b_bins=12):
    n_bins = l_bins*a_bins*b_bins
    index_mat = np.zeros((l_bins,a_bins,b_bins,3))
    for idx in range(l_bins):
        for jdx in range(a_bins):
            for kdx in range(b_bins):
                index_mat[idx,jdx,kdx] = np.array([idx,jdx,kdx])
    index_mat = index_mat.reshape(n_bins,3)
    all_dists = euclidean_distances(index_mat,index_mat)
    return all_dists / np.max(all_dists)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--opacityImageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--rgbImageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--jitter', type=int, default=0, help='no jitter')
parser.add_argument('--sequential', action='store_true', help='sequentially sample data')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gid'  , type=int, default=0, help='GPU id')
parser.add_argument('--translateNets', nargs='*', help="path to translation gNets")
parser.add_argument('--err_filename', required=True, default='errs', help='error filename (npy)')
parser.add_argument('--num_workers', default='3', help='number of threads to help with data io')
parser.add_argument('--outf', default='', help='folder to output images')
parser.add_argument('--name', default='translation', help='name of output images')
parser.add_argument('--num_samples',default=-1,type=int,help="number of samples")

parser.set_defaults(cuda=True)
parser.set_defaults(sequential=True)

opt = parser.parse_args()
print(opt)

save_images = opt.outf is not ''
if save_images:
    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)
    if opt.outf[-1] != '/':
        opt.outf += '/'
    print(opt.outf)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

# data
dataset = TranslationDataset(opt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.num_workers))

# minibatch data
if opt.cuda:
    torch.cuda.device(opt.gid)
    input_view = Variable(torch.FloatTensor(opt.batchSize,5).cuda(), volatile=True)
    input_opacity_tf = Variable(torch.FloatTensor(opt.batchSize,1,256).cuda(), volatile=True)
    input_color_tf = Variable(torch.FloatTensor(opt.batchSize,4,256).cuda(), volatile=True)
    input_img = Variable(torch.FloatTensor(opt.batchSize,1,opt.rgbImageSize,opt.rgbImageSize).cuda())
else:
    input_view = Variable(torch.FloatTensor(opt.batchSize,5), volatile=True)
    input_opacity_tf = Variable(torch.FloatTensor(opt.batchSize,1,256), volatile=True)
    input_color_tf = Variable(torch.FloatTensor(opt.batchSize,4,256), volatile=True)
    input_img = Variable(torch.FloatTensor(opt.batchSize,1,opt.rgbImageSize,opt.rgbImageSize))
#

img_res = 3*opt.rgbImageSize*opt.rgbImageSize
emd_cost_mat = compute_emd_cost_mat()

all_rmse_errs = []
all_rel_errs = []
all_ssim = []
all_emd = []

for translation_net_filename in opt.translateNets:
    if opt.cuda:
        translationNet = torch.load(translation_net_filename,map_location={'cuda:0': 'cuda:%d'%opt.gid, 'cuda:1': 'cuda:%d'%opt.gid})
    else:
        translationNet = torch.load(translation_net_filename,map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu'})
    translationNet.eval()
    translationNet.opNet[0].eval()
    if opt.cuda:
        translationNet.cuda()

    net_rmse = []
    net_rel_error = []
    net_ssim = []
    net_emd = []
    for bdx, data in tqdm(enumerate(dataloader)):
        # grab minibatch
        mb_view,mb_opacity,mb_color,mb_op_img,mb_img = data
        batch_size = mb_view.size(0)

        # copy minibatch
        input_view.data.resize_(mb_view.size()).copy_(mb_view)
        input_opacity_tf.data.resize_(mb_opacity.size()).copy_(mb_opacity)
        input_color_tf.data.resize_(mb_color.size()).copy_(mb_color)
        input_img.data.resize_(mb_img.size()).copy_(mb_img)

        # forward
        predicted_opacity,_ = translationNet.opNet[0]([input_view,input_opacity_tf])
        predicted_img = translationNet([input_view,input_opacity_tf,input_color_tf,predicted_opacity])

        true_img_batch = torch.Tensor.numpy(input_img.data.cpu())
        predicted_img_batch = torch.Tensor.numpy(predicted_img.data.cpu())
        for idx in range(true_img_batch.shape[0]):
            true_img = (0.5*(true_img_batch[idx,:,:,:]+1.0))
            predicted_img = (0.5*(predicted_img_batch[idx,:,:,:]+1.0))
            rmse = np.sqrt(np.sum((true_img-predicted_img)**2)/img_res)
            rel_error = np.linalg.norm(true_img-predicted_img)/np.linalg.norm(true_img)
            net_rmse.append(rmse)
            net_rel_error.append(rel_error)

            diff_img = np.abs(true_img-predicted_img)
            true_img *= 255
            predicted_img *= 255
            diff_img *= 255

            predicted_img = predicted_img.astype(np.uint8)
            predicted_rgb_img = np.zeros((opt.rgbImageSize,opt.rgbImageSize,3),dtype=np.uint8)
            true_img = true_img.astype(np.uint8)
            true_rgb_img = np.zeros((opt.rgbImageSize,opt.rgbImageSize,3),dtype=np.uint8)
            diff_img = diff_img.astype(np.uint8)
            diff_rgb_img = np.zeros((opt.rgbImageSize,opt.rgbImageSize,3),dtype=np.uint8)
            for i in range(3):
                predicted_rgb_img[:,:,i] = predicted_img[i,:,:]
                true_rgb_img[:,:,i] = true_img[i,:,:]
                diff_rgb_img[:,:,i] = diff_img[i,:,:]

            predicted_pil_img = Image.fromarray(predicted_rgb_img)
            if save_images:
                predicted_pil_img.save(opt.outf+opt.name+'-'+str(bdx)+'-'+str(idx)+'.png')
            true_pil_img = Image.fromarray(true_rgb_img)
            if save_images:
                true_pil_img.save(opt.outf+opt.name+'-'+str(bdx)+'-'+str(idx)+'-gt.png')

            # ssim
            ssim_val = compute_ssim(true_pil_img,predicted_pil_img)
            net_ssim.append(ssim_val)

            # emd
            emd_val = compute_emd(predicted_rgb_img,true_rgb_img,emd_cost_mat)
            net_emd.append(emd_val)

            if save_images:
                Image.fromarray(diff_rgb_img).save(opt.outf+opt.name+'-'+str(bdx)+'-'+str(idx)+'-diff.png')

    #
    all_rmse_errs.append(net_rmse)
    all_rel_errs.append(net_rel_error)
    all_ssim.append(net_ssim)
    all_emd.append(net_emd)
#

all_rmse_errs = np.array(all_rmse_errs)
all_rel_errs = np.array(all_rel_errs)
all_ssim = np.array(all_ssim)
all_emd = np.array(all_emd)
print('average RMSEs:',np.mean(all_rmse_errs,axis=1),'SSIM:',np.mean(all_ssim,axis=1),'EMD:',np.mean(all_emd,axis=1))

all_errs = np.zeros((4,all_rel_errs.shape[0],all_rel_errs.shape[1]))
all_errs[0,:,:] = all_rmse_errs
all_errs[1,:,:] = all_rel_errs
all_errs[2,:,:] = all_ssim
all_errs[3,:,:] = all_emd

np.save(opt.err_filename,all_errs)
