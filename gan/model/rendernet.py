from __future__ import print_function
import argparse
import os
import collections
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np

from model.res_utils import Basic1DResBlock, BasicResBlock, BottleneckResBlock

# opacity network

class OpacityNetGeneratorAE(nn.Module):
    def __init__(self, opt):
        super(OpacityNetGeneratorAE, self).__init__()
        self.res = opt.imageSize
        self.nvf = opt.nvf
        self.nof = opt.nof
        self.ngf = opt.ngf

        self.max_channels = opt.max_channels
        self.min_spatial_res = 2
        self.latent_dim = opt.latent_dim

        self.linear_bias = True
        self.conv1d_bias = True
        self.conv2d_bias = True

        # figure out feature map nonsense
        all_nplanes = [1]
        res = self.min_spatial_res / 2
        res_mult = 1
        while res != self.res:
            next_nplanes = min(self.ngf*res_mult,self.max_channels)
            all_nplanes.insert(0,next_nplanes)
            res_mult *= 2
            res*=2

        # view network
        self.view_subnet = nn.Sequential(
            nn.Linear(5,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nvf,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True)
        )

        # opacity transfer function network
        self.opacity_encoder = nn.Sequential(
            nn.Conv1d(1,16,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 128
            nn.Conv1d(16,32,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 64
            nn.Conv1d(32,64,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 32
            nn.Conv1d(64,128,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 16
            nn.Conv1d(128,128,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 8
            nn.Conv1d(128,256,8, stride=1, padding=0, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True) # 1
        )
        self.opacity_latent_encoder = nn.Sequential(nn.Linear(256,self.latent_dim,bias=self.linear_bias),nn.LeakyReLU(0.2,inplace=True))
        self.opacity_latent_decoder = nn.Sequential(nn.Linear(self.latent_dim,self.nof,bias=self.linear_bias),nn.LeakyReLU(0.2,inplace=True))
        self.opacity_latent_reconstructor = nn.Sequential(
            nn.Linear(self.latent_dim,128,bias=self.linear_bias),nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,256,bias=self.linear_bias),nn.Tanh()
        )

        # merged subnet
        self.merged_subnet = nn.Sequential(
            nn.Linear((self.nvf+self.nof),all_nplanes[0],bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(all_nplanes[0],all_nplanes[0],bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True)
        )

        # decoder
        cur_res = self.min_spatial_res / 2
        rdx = 0
        decoder_list = []
        while cur_res != self.res:
            cur_res *= 2
            in_planes = all_nplanes[rdx]
            out_planes = all_nplanes[rdx+1]
            kernel_size = 1 if cur_res == self.min_spatial_res else 3
            padding = 0 if cur_res == self.min_spatial_res else 1

            print('residual encoder[',cur_res,']: in planes:',in_planes,'out planes:',out_planes,'num params:',(2*in_planes*out_planes*kernel_size*kernel_size + in_planes*out_planes))
            decoder_list.append(('upsample'+str(rdx),nn.UpsamplingNearest2d(scale_factor=2)))
            decoder_list.append(('resblock'+str(rdx),BasicResBlock(in_planes,out_planes,kernel_size,1,padding,do_activation=cur_res!=self.res)))
            if cur_res == self.res:
                decoder_list.append(('tanh',nn.Tanh()))
            rdx+=1
        self.decoder_subnet = nn.Sequential(collections.OrderedDict(decoder_list))
    #

    def encode_opacity(self,tf):
        b_size = tf.size()[0]
        opacity_encoded = self.opacity_encoder(tf).view(b_size,256)
        return self.opacity_latent_encoder(opacity_encoded)

    def forward(self, input):
        b_size = input[0].size()[0]
        # feed in view
        view_out = self.view_subnet(input[0])
        # feed in opacity
        opacity_encoded = self.opacity_encoder(input[1]).view(b_size,256)
        latent_opacity_code = self.opacity_latent_encoder(opacity_encoded)
        opacity_encoded = self.opacity_latent_decoder(latent_opacity_code)

        # reshape opacity and merge with view
        joint_view_opacity = torch.cat((view_out,opacity_encoded),1)

        # feed in to merged subnet
        joint_view_opacity_out = self.merged_subnet(joint_view_opacity)

        # reshape into 4-tensor
        img_encoding = joint_view_opacity_out.view(b_size,joint_view_opacity_out.size()[1],1,1)

        # feed into the decoder to get the opacity image
        predicted_img = self.decoder_subnet(img_encoding)

        # and predict op TF
        predicted_op_tf = self.opacity_latent_reconstructor(latent_opacity_code)
        predicted_op_tf = predicted_op_tf.view(b_size,1,256)

        return predicted_img,predicted_op_tf
    #
#

class OpacityNetGenerator(nn.Module):
    def __init__(self, opt):
        super(OpacityNetGenerator, self).__init__()
        self.res = opt.imageSize
        self.nvf = opt.nvf
        self.nof = opt.nof
        self.ngf = opt.ngf

        self.max_channels = opt.max_channels
        self.min_spatial_res = 2
        self.latent_dim = opt.latent_dim

        self.linear_bias = True
        self.conv1d_bias = True
        self.conv2d_bias = True

        # figure out feature map nonsense
        all_nplanes = [1]
        res = self.min_spatial_res / 2
        res_mult = 1
        while res != self.res:
            next_nplanes = min(self.ngf*res_mult,self.max_channels)
            all_nplanes.insert(0,next_nplanes)
            res_mult *= 2
            res*=2

        # view network
        self.view_subnet = nn.Sequential(
            nn.Linear(5,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nvf,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True)
        )

        # opacity transfer function network
        self.opacity_encoder = nn.Sequential(
            nn.Conv1d(1,16,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 128
            nn.Conv1d(16,32,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 64
            nn.Conv1d(32,64,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 32
            nn.Conv1d(64,128,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 16
            nn.Conv1d(128,128,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 8
            nn.Conv1d(128,256,8, stride=1, padding=0, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True) # 1
        )
        self.opacity_latent_encoder = nn.Sequential(nn.Linear(256,self.latent_dim,bias=self.linear_bias),nn.LeakyReLU(0.2,inplace=True))
        self.opacity_latent_decoder = nn.Sequential(nn.Linear(self.latent_dim,self.nof,bias=self.linear_bias),nn.LeakyReLU(0.2,inplace=True))

        # merged subnet
        self.merged_subnet = nn.Sequential(
            nn.Linear((self.nvf+self.nof),all_nplanes[0],bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(all_nplanes[0],all_nplanes[0],bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True)
        )

        # decoder
        cur_res = self.min_spatial_res / 2
        rdx = 0
        decoder_list = []
        while cur_res != self.res:
            cur_res *= 2
            in_planes = all_nplanes[rdx]
            out_planes = all_nplanes[rdx+1]
            kernel_size = 1 if cur_res == self.min_spatial_res else 3
            padding = 0 if cur_res == self.min_spatial_res else 1

            print('residual encoder[',cur_res,']: in planes:',in_planes,'out planes:',out_planes,'num params:',(2*in_planes*out_planes*kernel_size*kernel_size + in_planes*out_planes))
            decoder_list.append(('upsample'+str(rdx),nn.UpsamplingNearest2d(scale_factor=2)))
            decoder_list.append(('resblock'+str(rdx),BasicResBlock(in_planes,out_planes,kernel_size,1,padding,do_activation=cur_res!=self.res)))
            if cur_res == self.res:
                decoder_list.append(('tanh',nn.Tanh()))
            rdx+=1
        self.decoder_subnet = nn.Sequential(collections.OrderedDict(decoder_list))
    #

    def encode_opacity(self,tf):
        b_size = tf.size()[0]
        opacity_encoded = self.opacity_encoder(tf).view(b_size,256)
        return self.opacity_latent_encoder(opacity_encoded)

    def forward(self, input):
        b_size = input[0].size()[0]
        # feed in view
        view_out = self.view_subnet(input[0])
        # feed in opacity
        opacity_encoded = self.opacity_encoder(input[1]).view(b_size,256)
        latent_opacity_code = self.opacity_latent_encoder(opacity_encoded)
        opacity_encoded = self.opacity_latent_decoder(latent_opacity_code)

        # reshape opacity and merge with view
        joint_view_opacity = torch.cat((view_out,opacity_encoded),1)

        # feed in to merged subnet
        joint_view_opacity_out = self.merged_subnet(joint_view_opacity)

        # reshape into 4-tensor
        img_encoding = joint_view_opacity_out.view(b_size,joint_view_opacity_out.size()[1],1,1)

        # feed into the decoder to get the opacity image
        predicted_img = self.decoder_subnet(img_encoding)

        return predicted_img
    #
#

class OpacityNetDiscriminator(nn.Module):
    def __init__(self, opt):
        super(OpacityNetDiscriminator, self).__init__()
        self.res = opt.imageSize
        self.nvf = opt.nvf
        self.nof = opt.nof
        self.ndf = opt.ndf

        self.max_channels = opt.max_channels
        self.min_spatial_res = opt.min_spatial_res
        self.do_regularization = False

        self.linear_bias = True
        self.conv1d_bias = True
        self.conv2d_bias = True

        # figure out feature map nonsense
        all_nplanes = [1]
        res = self.min_spatial_res
        res_mult = 1
        while res != self.res:
            next_nplanes = min(self.ndf*res_mult,self.max_channels)
            all_nplanes.append(next_nplanes)
            res_mult *= 2
            res*=2

        # view network
        self.view_subnet = nn.Sequential(
            nn.Linear(5,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nvf,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True)
        )

        # opacity transfer function network
        self.opacity_encoder = nn.Sequential(
            nn.Conv1d(1,16,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 128
            nn.Conv1d(16,32,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 64
            nn.Conv1d(32,64,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 32
            nn.Conv1d(64,128,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 16
            nn.Conv1d(128,256,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 8
            nn.Conv1d(256,self.nof,8, stride=1, padding=0, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True) # 1
        )

        # image encoder
        encoder_list = []
        cur_res = self.min_spatial_res
        rdx = 0
        while cur_res != self.res:
            cur_res *= 2
            kernel_size,stride,padding=4,2,1
            if cur_res == self.res:
                kernel_size=2*self.min_spatial_res
                stride=1
                padding=0

            print('[disc] encoder',rdx,'in planes:',all_nplanes[rdx],'out planes:',all_nplanes[rdx+1],'resolution:',cur_res)
            encoder_list.append(('conv'+str(rdx),nn.Conv2d(all_nplanes[rdx],all_nplanes[rdx+1],kernel_size,stride,padding)))
            if self.do_regularization:
                encoder_list.append(('selu'+str(rdx),nn.SELU(inplace=True)))
            else:
                if cur_res != (2*self.min_spatial_res) and cur_res != self.res:
                    encoder_list.append(('bn'+str(rdx),nn.BatchNorm2d(all_nplanes[rdx+1])))
                encoder_list.append(('relu'+str(rdx),nn.LeakyReLU(0.2,inplace=True)))
            rdx+=1
        #
        self.encoder_subnet = nn.Sequential(collections.OrderedDict(encoder_list))

        # merge everything, some FC layers, then take sigmoid
        self.decision_subnet = nn.Sequential(
            nn.Linear((self.nvf+self.nof+all_nplanes[-1]),all_nplanes[-1],bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(all_nplanes[-1],all_nplanes[-1],bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(all_nplanes[-1],1,bias=self.linear_bias))
        self.sigmoid = nn.Sigmoid()
    #

    def forward(self, input):
        b_size = input[0].size()[0]
        # feed in view
        view_out = self.view_subnet(input[0])
        # feed in opacity
        opacity_encoded = self.opacity_encoder(input[1]).view(b_size,self.nof)
        # feed in image
        image_out = self.encoder_subnet(input[2])

        # reshape opacity and image, and merge with view
        image_out = image_out.view(b_size, image_out.size()[1])
        joint_encoding = torch.cat((view_out,opacity_encoded,image_out),1)

        # fake or not?
        raw_decision = self.decision_subnet(joint_encoding)
        decision = self.sigmoid(raw_decision)

        return decision,raw_decision
    #
#

# translation network

class TranslationNetGenerator(nn.Module):
    def __init__(self, opt):
        super(TranslationNetGenerator, self).__init__()
        self.opacity_res = opt.opacityImageSize
        self.rgb_res = opt.rgbImageSize
        self.nvf = opt.nvf
        self.ntf = opt.ntf
        self.ngf = opt.ngf

        self.feat_multiplier = 2
        self.max_channels = opt.max_channels
        self.min_spatial_res = opt.min_spatial_res
        self.nbottleneck = opt.nbottleneck

        self.do_rb_upsampling = opt.rbupsample
        self.upsample_channels = 32 if self.do_rb_upsampling else 64
        self.do_upsample = self.opacity_res != self.rgb_res
        self.latent_color_dim = 32
        self.use_latent_opacity = opt.opNet != ''
        self.tile_tfs = False

        self.use_noise = opt.noise_dim > 0

        self.linear_bias = True
        self.conv1d_bias = True
        self.conv2d_bias = True

        self.debug = False

        if self.use_latent_opacity:
            self.opNet = [torch.load(opt.opNet,map_location={'cuda:0': 'cuda:%d'%opt.gid, 'cuda:1': 'cuda:%d'%opt.gid})]
            self.opNet[0].cuda()
            self.opNet[0].eval()
            self.latent_op_dim = self.opNet[0].latent_dim
        else:
            self.latent_op_dim = 32

        # encoder/decoder spatial resolutions, and feature map resolutions
        self.encoder_spatial_resolutions = [self.opacity_res]
        self.encoder_nplanes = [1]
        spatial_res = self.opacity_res
        feat_res_mult = 1
        while spatial_res != self.min_spatial_res:
            spatial_res /= 2
            self.encoder_spatial_resolutions.append(spatial_res)
            next_nplanes = min(int(self.ngf*feat_res_mult),self.max_channels)
            self.encoder_nplanes.append(next_nplanes)
            feat_res_mult *= self.feat_multiplier
        self.encoder_spatial_resolutions = np.array(self.encoder_spatial_resolutions)
        self.encoder_nplanes = np.array(self.encoder_nplanes)
        self.decoder_spatial_resolutions = np.flip(self.encoder_spatial_resolutions,0)
        self.decoder_nplanes = np.copy(np.flip(self.encoder_nplanes,0))
        self.decoder_nplanes[-1] = self.ngf if self.do_upsample else 3

        self.n_encoder_layers = self.encoder_spatial_resolutions.shape[0]-1
        self.n_decoder_layers = self.n_encoder_layers

        # view network
        self.view_encoder = nn.Sequential(
            nn.Linear(5,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nvf,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True)
        )

        # opacity TF network
        if not self.use_latent_opacity:
            self.opacity_encoder = nn.Sequential(
                nn.Conv1d(1,16,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 128
                nn.Conv1d(16,32,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 64
                nn.Conv1d(32,64,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 32
                nn.Conv1d(64,128,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 16
                nn.Conv1d(128,128,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 8
                nn.Conv1d(128,self.ntf,8, stride=1, padding=0, bias=False), nn.LeakyReLU(0.2, inplace=True) # 1
            )
            self.opacity_latent_encoder = nn.Sequential(nn.Linear(self.ntf,self.latent_op_dim,bias=self.linear_bias),nn.LeakyReLU(0.2,inplace=True))
        #
        self.opacity_latent_decoder = nn.Sequential(nn.Linear(self.latent_op_dim,self.ntf,bias=self.linear_bias),nn.LeakyReLU(0.2,inplace=True))
        # color TF network
        self.color_encoder = nn.Sequential(
            nn.Conv1d(3,16,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 128
            nn.Conv1d(16,32,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 64
            nn.Conv1d(32,64,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 32
            nn.Conv1d(64,128,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 16
            nn.Conv1d(128,128,5, stride=2, padding=2, bias=False), nn.LeakyReLU(0.2, inplace=True), # 8
            nn.Conv1d(128,self.ntf,8, stride=1, padding=0, bias=False), nn.LeakyReLU(0.2, inplace=True) # 1
        )
        self.color_latent_encoder = nn.Sequential(nn.Linear(self.ntf,self.latent_color_dim,bias=self.linear_bias),nn.LeakyReLU(0.2,inplace=True))
        self.color_latent_decoder = nn.Sequential(nn.Linear(self.latent_color_dim,self.ntf,bias=self.linear_bias),nn.LeakyReLU(0.2,inplace=True))

        # image encoder
        self.image_encoder_layers = nn.ModuleList()
        for rdx in range(self.n_encoder_layers):
            in_planes,out_planes = int(self.encoder_nplanes[rdx]),int(self.encoder_nplanes[rdx+1])
            encoder_layer_list = []
            if rdx == 0:
                encoder_layer_list.append(('conv',nn.Conv2d(in_planes,out_planes,4,2,1)))
            else:
                encoder_layer_list.append(('conv',nn.Conv2d(in_planes,out_planes,4,2,1)))
                encoder_layer_list.append(('bn',nn.BatchNorm2d(out_planes)))
            encoder_layer_list.append(('relu',nn.LeakyReLU(0.2,inplace=True)))
            self.image_encoder_layers.append(nn.Sequential(collections.OrderedDict(encoder_layer_list)))
        #

        n_vis_params = 2*self.ntf+self.nvf+opt.noise_dim
        print('n vis params:',n_vis_params,'image resolution:',(self.encoder_nplanes[-1]*self.min_spatial_res*self.min_spatial_res))

        # now, this will be the opacity, color, and view feature maps appended onto the image feature maps -> apply bottleneck resnets to fuse features
        self.bottleneck_residual_layers = nn.ModuleList()
        fused_in_planes = int(self.encoder_nplanes[-1])+n_vis_params
        for bdx in range(self.nbottleneck):
            out_planes = int(self.encoder_nplanes[-1]) if bdx==self.nbottleneck-1 else fused_in_planes
            self.bottleneck_residual_layers.append(BottleneckResBlock(fused_in_planes,out_planes,bottleneck_scale=2))

        # decoder with skip connections: upsample, resblock, you know the drill
        self.image_decoder_layers = nn.ModuleList()
        for ddx in range(self.n_decoder_layers):
            in_planes = int(self.decoder_nplanes[ddx]) if ddx==0 else int(2*self.decoder_nplanes[ddx])
            out_planes = int(self.decoder_nplanes[ddx]) if ddx==self.n_decoder_layers-1 else int(self.decoder_nplanes[ddx+1])
            do_activation = ddx!=(self.n_decoder_layers-1) or self.do_upsample
            print('decoder',ddx,in_planes,'->',self.decoder_nplanes[ddx+1])
            decoder_layer_list = []
            decoder_layer_list.append(('upsample'+str(ddx),nn.UpsamplingNearest2d(scale_factor=2)))
            if self.do_rb_upsampling:
                decoder_layer_list.append(('resblock'+str(ddx),BasicResBlock(in_planes,int(self.decoder_nplanes[ddx+1]),do_activation=do_activation)))
            else:
                decoder_layer_list.append(('resblock'+str(ddx),BasicResBlock(in_planes,out_planes)))
                decoder_layer_list.append(('resblock-refine'+str(ddx),BasicResBlock(out_planes,int(self.decoder_nplanes[ddx+1]),do_activation=do_activation)))
            if not do_activation:
                decoder_layer_list.append(('tanh',nn.Tanh()))
            self.image_decoder_layers.append(nn.Sequential(collections.OrderedDict(decoder_layer_list)))

        # optionally, if opacity and rgb are of different resolution, then upsample until we good
        if self.do_upsample:
            cur_res = self.opacity_res
            self.upsample_layers = nn.ModuleList()
            while cur_res != self.rgb_res:
                cur_res *= 2
                in_planes = self.ngf if cur_res==(2*self.opacity_res) else self.upsample_channels
                out_planes = 3 if cur_res==self.rgb_res else self.upsample_channels
                print('upsample decoder',cur_res,in_planes,'->',out_planes)
                do_activation = cur_res != self.rgb_res
                upsample_layer_list = []
                upsample_layer_list.append(('upsample'+str(cur_res),nn.UpsamplingNearest2d(scale_factor=2)))
                if self.do_rb_upsampling:
                    if do_activation:
                        upsample_layer_list.append(('upsample-resblock'+str(cur_res),BasicResBlock(in_planes,out_planes,do_activation=do_activation)))
                    else:
                        upsample_layer_list.append(('upsample-conv'+str(cur_res),nn.Conv2d(in_planes,in_planes,3,1,1)))
                        upsample_layer_list.append(('upsample-bn'+str(cur_res),nn.BatchNorm2d(in_planes)))
                        upsample_layer_list.append(('upsample-relu'+str(cur_res),nn.ReLU(inplace=True)))
                        upsample_layer_list.append(('final-conv'+str(cur_res),nn.Conv2d(in_planes,out_planes,3,1,1)))
                else:
                    upsample_layer_list.append(('upsample-conv'+str(cur_res),nn.Conv2d(in_planes,out_planes,3,1,1)))
                    if do_activation:
                        upsample_layer_list.append(('upsample-bn'+str(cur_res),nn.BatchNorm2d(out_planes)))
                        upsample_layer_list.append(('upsample-relu'+str(cur_res),nn.ReLU(inplace=True)))
                if not do_activation:
                    upsample_layer_list.append(('tanh',nn.Tanh()))
                self.upsample_layers.append(nn.Sequential(collections.OrderedDict(upsample_layer_list)))
    #

    def set_opacity_net(self, opnet_filename, use_cuda=True):
        self.opNet = [torch.load(opnet_filename)]
        if use_cuda:
            self.opNet[0].cuda()
        self.opNet[0].eval()

    def directly_set_opacity_net(self, opnet):
        self.opNet = opnet

    def forward(self, input):
        b_size = input[0].size()[0]

        # encode view
        view_encoded = self.view_encoder(input[0])

        # encode TF
        if self.use_latent_opacity:
            latent_opacity = self.opNet[0].encode_opacity(input[1]).detach()
            opacity_encoded = self.opacity_latent_decoder(latent_opacity)
        else:
            opacity_encoded = self.opacity_encoder(input[1]).view(b_size,self.ntf)
            latent_opacity_code = self.opacity_latent_encoder(opacity_encoded)
            opacity_encoded = self.opacity_latent_decoder(latent_opacity_code)
        '''
        opacity_encoded = self.opacity_encoder(input[1]).view(b_size,self.ntf)
        latent_opacity_code = self.opacity_latent_encoder(opacity_encoded)
        opacity_encoded = self.opacity_latent_decoder(latent_opacity_code)
        '''

        color_encoded = self.color_encoder(input[2]).view(b_size,self.ntf)
        latent_color_code = self.color_latent_encoder(color_encoded)
        color_encoded = self.color_latent_decoder(latent_color_code)

        # feed in image
        image_encoded = input[3]
        all_encoded_images = []
        for image_layer in self.image_encoder_layers:
            image_encoded = image_layer(image_encoded)
            all_encoded_images.append(image_encoded)

        # merge the view, opacity, and color TFs
        #joint_vis_encoding = torch.cat((view_encoded,opacity_encoded,color_encoded),1)
        if self.use_noise:
            joint_vis_encoding = torch.cat((view_encoded,opacity_encoded,color_encoded,input[4]),1)
        else:
            joint_vis_encoding = torch.cat((view_encoded,opacity_encoded,color_encoded),1)
        channel_size = joint_vis_encoding.size()[1]
        tiled_encoding = joint_vis_encoding.view(b_size, channel_size, 1, 1)
        tiled_encoding = tiled_encoding.expand(b_size, channel_size, self.min_spatial_res, self.min_spatial_res)

        # merge image with tiled vis encoding
        full_code = torch.cat((image_encoded,tiled_encoding),1)

        # push through bottleneck layers to fuse features
        fused_encoding = full_code
        for layer in self.bottleneck_residual_layers:
            if self.debug:
                print('bottleneck layer:',layer,'input size:',fused_encoding.size())
            fused_encoding = layer(fused_encoding)

        # decode the image, adjoining the feature maps from the image encoding
        decoded_image = fused_encoding
        ddx = 0
        cur_res = self.min_spatial_res
        for layer,encoded_image in zip(self.image_decoder_layers,reversed(all_encoded_images)):
            joint_encoding = decoded_image if ddx == 0 else torch.cat((decoded_image,encoded_image),1)
            if self.debug:
                print('decoding layer:',layer)
            decoded_image = layer(joint_encoding)
            ddx+=1
            cur_res *= 2

        # optionally upsample
        if self.do_upsample:
            for layer in self.upsample_layers:
                if self.debug:
                    print('decoding layer:',layer)
                decoded_image = layer(decoded_image)

        return decoded_image
    #
#

class TranslationNetDiscriminator(nn.Module):
    def __init__(self, opt):
        super(TranslationNetDiscriminator, self).__init__()
        self.opacity_res = opt.opacityImageSize
        self.rgb_res = opt.rgbImageSize
        self.nvf = opt.nvf
        self.ntf = opt.ntf
        self.ndf = opt.ndf

        self.feat_multiplier = 2
        self.max_channels = opt.max_channels
        self.min_spatial_res = min(4,opt.min_spatial_res)
        self.nbottleneck = opt.nbottleneck
        self.do_regularization = False

        self.linear_bias = True
        self.conv1d_bias = True
        self.conv2d_bias = True

        self.debug = False

        self.upsample_channels = 32
        self.do_upsample = self.opacity_res != self.rgb_res

        # encoder/decoder spatial resolutions, and feature map resolutions
        self.encoder_spatial_resolutions = [self.opacity_res]
        self.encoder_nplanes = [self.ndf] if self.do_upsample else [3]
        spatial_res = self.opacity_res
        feat_res_mult = 2 if self.do_upsample else 1
        while spatial_res >= self.min_spatial_res:
            spatial_res /= 2
            self.encoder_spatial_resolutions.append(spatial_res)
            next_nplanes = min(int(self.ndf*feat_res_mult),self.max_channels)
            self.encoder_nplanes.append(next_nplanes)
            feat_res_mult *= self.feat_multiplier
        self.encoder_spatial_resolutions = np.array(self.encoder_spatial_resolutions)
        self.encoder_nplanes = np.array(self.encoder_nplanes)

        self.n_encoder_layers = self.encoder_spatial_resolutions.shape[0]-1

        # view network
        self.view_encoder = nn.Sequential(
            nn.Linear(5,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nvf,self.nvf,bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True)
        )

        self.opacity_encoder = nn.Sequential(
            nn.Conv1d(1,16,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 128
            nn.Conv1d(16,32,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 64
            nn.Conv1d(32,64,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 32
            nn.Conv1d(64,128,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 16
            nn.Conv1d(128,128,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 8
            nn.Conv1d(128,self.ntf,8, stride=1, padding=0, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True) # 1
        )
        self.color_encoder = nn.Sequential(
            nn.Conv1d(3,16,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 128
            nn.Conv1d(16,32,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 64
            nn.Conv1d(32,64,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 32
            nn.Conv1d(64,128,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 16
            nn.Conv1d(128,128,5, stride=2, padding=2, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True), # 8
            nn.Conv1d(128,self.ntf,8, stride=1, padding=0, bias=self.conv1d_bias), nn.LeakyReLU(0.2, inplace=True) # 1
        )

        # image encoder
        self.image_encoder_layers = nn.ModuleList()
        for rdx in range(self.n_encoder_layers):
            in_planes,out_planes = int(self.encoder_nplanes[rdx]),int(self.encoder_nplanes[rdx+1])
            kernel_size,stride,padding=4,2,1
            if rdx == self.n_encoder_layers-1:
                kernel_size=self.min_spatial_res
                stride=1
                padding=0
            print('layer[',rdx,']: channels',in_planes,out_planes,'; kernel size:',kernel_size,'stride:',stride)
            self.image_encoder_layers.append(nn.Conv2d(in_planes,out_planes,kernel_size,stride,padding))
            if self.do_regularization:
                self.image_encoder_layers.append(nn.SELU(inplace=True))
            else:
                if rdx != (self.n_encoder_layers-1) and (rdx > 0 or self.do_upsample):
                    self.image_encoder_layers.append(nn.BatchNorm2d(out_planes))
                self.image_encoder_layers.append(nn.LeakyReLU(0.2,inplace=True))
        #

        # if we are upsampling, then a bit more involved: need to perform some convolutions w/ strides on rgb image
        if self.do_upsample:
            self.pre_rgb_encoder = nn.ModuleList()
            cur_res = self.opacity_res
            while cur_res != self.rgb_res:
                cur_res *= 2
                in_planes = 3 if cur_res==(2*self.opacity_res) else self.upsample_channels
                out_planes = self.ndf if cur_res==self.rgb_res else self.upsample_channels
                print('pre layer[',cur_res,']: channels',in_planes,out_planes)
                self.pre_rgb_encoder.append(nn.Conv2d(in_planes,out_planes,4,2,1))
                if self.do_regularization:
                    self.pre_rgb_encoder.append(nn.SELU(inplace=True))
                else:
                    if cur_res != (2*self.opacity_res):
                        self.pre_rgb_encoder.append(nn.BatchNorm2d(out_planes))
                    self.pre_rgb_encoder.append(nn.LeakyReLU(0.2,inplace=True))

        n_vis_params = 2*self.ntf+self.nvf
        # merge everything, some FC layers, then take sigmoid
        self.decision_subnet = nn.Sequential(
            nn.Linear((n_vis_params+int(self.encoder_nplanes[-1])),int(self.encoder_nplanes[-1]),bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.encoder_nplanes[-1]),int(self.encoder_nplanes[-1]),bias=self.linear_bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.encoder_nplanes[-1]),1,bias=self.linear_bias))
        self.sigmoid = nn.Sigmoid()
    #

    def forward(self, input):
        b_size = input[0].size()[0]

        # encode view
        view_encoded = self.view_encoder(input[0])

        # encode TF
        opacity_encoded = self.opacity_encoder(input[1]).view(b_size,self.ntf)
        color_encoded = self.color_encoder(input[2]).view(b_size,self.ntf)

        # if we are upsampling, then first process rgb image, then merge with opacity, proceed to encoding
        if self.do_upsample:
            rgb_encoded = input[3]
            for layer in self.pre_rgb_encoder:
                rgb_encoded = layer(rgb_encoded)
            image_encoded = rgb_encoded
        else:
            image_encoded = input[3]
        for image_layer in self.image_encoder_layers:
            if self.debug:
                print('image layer:',image_layer)
            image_encoded = image_layer(image_encoded)

        # reshape transfer functions and image, and merge with view
        image_encoded = image_encoded.view(b_size, image_encoded.size()[1])
        joint_encoding = torch.cat((view_encoded,opacity_encoded,color_encoded,image_encoded),1)

        # fake or not?
        raw_decision = self.decision_subnet(joint_encoding)
        decision = self.sigmoid(raw_decision)

        return decision,raw_decision
    #
#

def feature_forward(opNet, translationNet, input_view, encoded_op_tf_var, input_color_tf):
    b_size = input_view.size()[0]

    # feed in view
    view_out = opNet.view_subnet(input_view)
    # feed in opacity
    opacity_encoded = opNet.opacity_latent_decoder(encoded_op_tf_var)

    # reshape opacity and merge with view
    joint_view_opacity = torch.cat((view_out,opacity_encoded),1)

    # feed in to merged subnet
    joint_view_opacity_out = opNet.merged_subnet(joint_view_opacity)

    # reshape into 4-tensor
    img_encoding = joint_view_opacity_out.view(b_size,joint_view_opacity_out.size()[1],1,1)

    # feed into the decoder to get the opacity image
    predicted_op_img = opNet.decoder_subnet(img_encoding)

    # encode view
    view_encoded = translationNet.view_encoder(input_view)

    # encode TF
    opacity_encoded = translationNet.opacity_latent_decoder(encoded_op_tf_var)
    color_encoded = translationNet.color_encoder(input_color_tf).view(b_size,translationNet.ntf)
    latent_color_code = translationNet.color_latent_encoder(color_encoded)
    color_encoded = translationNet.color_latent_decoder(latent_color_code)

    # feed in image
    image_encoded = predicted_op_img
    all_encoded_images = []
    for image_layer in translationNet.image_encoder_layers:
        image_encoded = image_layer(image_encoded)
        all_encoded_images.append(image_encoded)

    # merge the view, opacity, and color TFs
    joint_vis_encoding = torch.cat((view_encoded,opacity_encoded,color_encoded),1)
    channel_size = joint_vis_encoding.size()[1]
    tiled_encoding = joint_vis_encoding.view(b_size, channel_size, 1, 1)
    tiled_encoding = tiled_encoding.expand(b_size, channel_size, translationNet.min_spatial_res, translationNet.min_spatial_res)

    # merge image with tiled vis encoding
    full_code = torch.cat((image_encoded,tiled_encoding),1)

    # push through bottleneck layers to fuse features
    fused_encoding = full_code
    for layer in translationNet.bottleneck_residual_layers:
        fused_encoding = layer(fused_encoding)

    # decode the image, adjoining the feature maps from the image encoding
    decoded_image = fused_encoding
    ddx = 0
    cur_res = translationNet.min_spatial_res
    for layer,encoded_image in zip(translationNet.image_decoder_layers,reversed(all_encoded_images)):
        joint_encoding = decoded_image if ddx == 0 else torch.cat((decoded_image,encoded_image),1)
        decoded_image = layer(joint_encoding)
        ddx+=1
        cur_res *= 2

    # optionally upsample
    if translationNet.do_upsample:
        for layer in translationNet.upsample_layers:
            decoded_image = layer(decoded_image)

    return decoded_image
