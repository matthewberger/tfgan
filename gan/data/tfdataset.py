import time
import numpy as np
import PIL
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset

class TFDataset(Dataset):
    def __init__(self, opt, n_channels=1, train_size=-1):
        self.base_dir=opt.dataroot
        if self.base_dir[-1] != '/':
            self.base_dir+='/'
        self.mb_size = opt.batchSize
        self.target_res = opt.imageSize
        self.n_channels = n_channels
        self.train_size = -1

        # base directory for images
        self.imgs_dir = self.base_dir+'imgs/'
        self.inputs_dir = self.base_dir+'inputs/'

        # all image files
        self.img_files_filename = self.base_dir+'files.csv'

        # all input files
        self.input_files_filename = self.base_dir+'inputs.csv'

        # load in filenames
        self.img_files = []
        self.input_files = []
        fdx=0
        for img_file,input_file in zip(open(self.img_files_filename),open(self.input_files_filename)):
            self.img_files.append(img_file.rstrip())
            self.input_files.append(input_file.rstrip())
            fdx+=1
            if fdx==train_size:
                break
        self.train_size = len(self.img_files)

        self.minibatch_idx = 0
        self.epoch = 0

        # TODO various hard-coded facts about input
        self.min_elevation = 0.0
        self.max_elevation = 180.0
        self.min_zoom = 1.0
        self.max_zoom = 2.5 # FIXME
        self.roll_range = 10.0
        self.tf_res = 256
        self.n_samples = opt.num_samples
    #

    def __len__(self):
        return self.train_size

    def grab_vis_params(self, index, get_op_img=False):
        with open(self.inputs_dir+self.input_files[index]) as input_file:
            all_inputs = np.array([float(val.rstrip()) for val in input_file])

        # setup view params
        view_params = np.zeros(5)
        view_params[0] = 2.0*((all_inputs[0]-self.min_elevation) / (self.max_elevation-self.min_elevation)) - 1.0
        view_params[1] = np.cos(np.deg2rad(all_inputs[1]))
        view_params[2] = np.sin(np.deg2rad(all_inputs[1]))
        view_params[3] = all_inputs[2] / self.roll_range
        view_params[4] = 2.0*((all_inputs[3]-self.min_zoom) / (self.max_zoom-self.min_zoom)) - 1.0

        # grab the opacity TF
        verbose_op_func = np.reshape(all_inputs[4:(4+2*self.tf_res)], (self.tf_res,2))
        op_tf = 2.0*verbose_op_func[:,1:] - 1.0
        op_tf = op_tf.T

        if get_op_img:
            pil_rgba_img = Image.open(self.imgs_dir+self.img_files[index]).convert("RGBA")
            np_rgba_img = np.array(pil_rgba_img).astype(np.uint8)
            np_opacity_img = np.zeros((np_rgba_img.shape[0],np_rgba_img.shape[1],3)).astype(np.uint8)
            np_opacity_img[:,:,0] = np_rgba_img[:,:,3]
            np_opacity_img[:,:,1] = np_rgba_img[:,:,3]
            np_opacity_img[:,:,2] = np_rgba_img[:,:,3]
            if self.target_res != pil_rgba_img.width:
                pil_opacity_img = Image.fromarray(np_opacity_img).resize((self.target_res,self.target_res), resample=PIL.Image.BICUBIC)
            else:
                pil_opacity_img = Image.fromarray(np_opacity_img)
            target_opacity_img = np.array(pil_opacity_img,dtype=np.float32)/255.0
            target_opacity_img = np.swapaxes(target_opacity_img[:,:,0:1].T,axis1=1,axis2=2)

            target_opacity_img = 2.0*target_opacity_img - 1.0

            return view_params,op_tf,target_opacity_img
        else:
            return view_params,op_tf

    def __getitem__(self, index):
        if self.n_samples != -1:
            index = index % self.n_samples
        np.random.seed(seed=int(time.time() + index))
        view_params,op_tf,target_opacity_img = self.grab_vis_params(index,True)

        return torch.from_numpy(view_params),torch.from_numpy(op_tf),torch.from_numpy(target_opacity_img)
    #
#
