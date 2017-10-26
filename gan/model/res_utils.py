from __future__ import print_function
import argparse
import os
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

class Basic1DResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, do_activation=True):
        super(Basic1DResBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.do_activation = do_activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.second_kernel_size = 1 if self.padding==0 else self.kernel_size

        self.conv1 = nn.Conv1d(in_planes,in_planes, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_planes,out_planes, kernel_size=self.second_kernel_size, stride=1, padding=self.padding, bias=True)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv1d(in_planes,out_planes, kernel_size=1, stride=self.stride, bias=True),
            nn.BatchNorm1d(out_planes)
        )
    #

    def forward(self, input):
        if self.in_planes != self.out_planes or self.stride > 1:
            residual = self.downsample(input)
        else:
            residual = input

        # push residual through first layer
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu1(out)

        # now the next
        out = self.conv2(out)
        out = self.bn2(out)

        # add it to the original input, ReLU, done
        out += residual
        if self.do_activation:
            out = self.relu2(out)

        return out
    #
#

# does not downsample, or mess with strides
class BasicResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, do_activation=True, do_initial_batchnorm=True, use_relu=True):
        super(BasicResBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.do_activation = do_activation
        self.do_initial_batchnorm = do_initial_batchnorm
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.second_kernel_size = 1 if self.padding==0 else self.kernel_size

        self.conv1 = nn.Conv2d(in_planes,in_planes, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        if use_relu:
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv2d(in_planes,out_planes, kernel_size=self.second_kernel_size, stride=self.stride, padding=self.padding, bias=False)
        if self.do_activation:
            self.bn2 = nn.BatchNorm2d(out_planes)
            if use_relu:
                self.relu2 = nn.ReLU(inplace=True)
            else:
                self.relu2 = nn.LeakyReLU(0.2,inplace=True)

        if self.in_planes != self.out_planes or self.stride > 1:
            if self.do_activation and self.do_initial_batchnorm:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes,out_planes, kernel_size=1, stride=self.stride, bias=False),
                    nn.BatchNorm2d(out_planes)
                )
            else:
                self.downsample = nn.Sequential(nn.Conv2d(in_planes,out_planes, kernel_size=1, stride=self.stride, bias=False))
    #

    def forward(self, input):
        #print('input shape:',input.size())
        if self.in_planes != self.out_planes or self.stride > 1:
            residual = self.downsample(input)
        else:
            residual = input

        # push residual through first layer
        out = self.conv1(input)
        #out = self.bn1(out)
        if self.do_initial_batchnorm:
            out = self.bn1(out)
        out = self.relu1(out)

        # now the next
        out = self.conv2(out)
        if self.do_activation:
            out = self.bn2(out)

        # add it to the original input, ReLU, done
        out += residual
        if self.do_activation:
            out = self.relu2(out)

        #out = residual
        return out
    #
#

class BottleneckResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, do_activation=True, bottleneck_scale=4):
        super(BottleneckResBlock, self).__init__()
        self.in_planes = in_planes
        self.lowres_in_planes = int(self.in_planes/bottleneck_scale)
        self.out_planes = out_planes
        self.lowres_out_planes = int(self.out_planes/bottleneck_scale)
        self.do_activation = do_activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(self.in_planes,self.lowres_in_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.lowres_in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.lowres_in_planes,self.lowres_out_planes, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm2d(self.lowres_out_planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.lowres_out_planes,self.out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        #self.conv3 = nn.Conv2d(self.lowres_out_planes,self.out_planes, kernel_size=self.kernel_size, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_planes)

        self.relu3 = nn.ReLU(inplace=True)

        self.need_downsampling = self.in_planes != self.out_planes or self.stride > 1
        if self.need_downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_planes,self.out_planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_planes)
            )
        else:
            self.downsample = None
    #

    def forward(self, input):
        if self.need_downsampling:
            residual = self.downsample(input)
        else:
            residual = input

        # push residual through first downsample layer
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu1(out)

        # bottleneck layer - 3x3 convs
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # push back to original layer
        out = self.conv3(out)
        out = self.bn3(out)

        # add it to the original input, ReLU, done
        out += residual
        if self.do_activation:
            out = self.relu3(out)

        return out
    #
#
