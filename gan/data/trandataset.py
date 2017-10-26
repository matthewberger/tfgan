import numpy as np
import time
import PIL
from PIL import Image

from colormath.color_objects import LabColor, sRGBColor, HSLColor
from colormath.color_conversions import convert_color

import torch
from torch.utils.data.dataset import Dataset

class TranslationDataset(Dataset):
    def __init__(self, opt, train_size=-1):
        self.base_dir=str(opt.dataroot)
        if self.base_dir[-1] != '/':
            self.base_dir+='/'
        self.mb_size = opt.batchSize
        self.target_opacity_res = opt.opacityImageSize
        self.target_rgb_res = opt.rgbImageSize
        self.jitter = opt.jitter
        self.train_size = train_size
        self.n_samples = opt.num_samples

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

        # TODO various hard-coded facts about input
        self.min_elevation = 0.0
        self.max_elevation = 180.0
        self.min_zoom = 1.0
        self.max_zoom = 2.5 # FIXME
        self.roll_range = 10.0
        self.tf_res = 256

        self.min_lab_l,self.max_lab_l = 0.0,100.0
        self.min_lab_a,self.max_lab_a = -86.185,98.254
        self.min_lab_b,self.max_lab_b = -107.863,94.482
    #

    def __len__(self):
        return self.train_size

    def grab_color_tf(self, index):
        with open(self.inputs_dir+self.input_files[index]) as input_file:
            all_inputs = np.array([float(val.rstrip()) for val in input_file])

        # grab the color TF
        verbose_color_func = np.reshape(all_inputs[(4+2*self.tf_res):], (self.tf_res,4))
        lab_color_tf = np.array([convert_color(sRGBColor(rgb[0],rgb[1],rgb[2]),LabColor).get_value_tuple() for rgb in verbose_color_func[:,1:]])
        color_tf = np.zeros((3,self.tf_res))
        color_tf[0,:] = 2.0*((lab_color_tf[:,0]-self.min_lab_l) / (self.max_lab_l-self.min_lab_l)) - 1.0
        color_tf[1,:] = 2.0*((lab_color_tf[:,1]-self.min_lab_a) / (self.max_lab_a-self.min_lab_a)) - 1.0
        color_tf[2,:] = 2.0*((lab_color_tf[:,2]-self.min_lab_b) / (self.max_lab_b-self.min_lab_b)) - 1.0

        return color_tf
    #

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

        # next, grab the opacity TF
        verbose_op_func = np.reshape(all_inputs[4:(4+2*self.tf_res)], (self.tf_res,2))
        op_tf = 2.0*verbose_op_func[:,1:] - 1.0
        op_tf = op_tf.T

        # grab the color TF
        verbose_color_func = np.reshape(all_inputs[(4+2*self.tf_res):], (self.tf_res,4))
        lab_color_tf = np.array([convert_color(sRGBColor(rgb[0],rgb[1],rgb[2]),LabColor).get_value_tuple() for rgb in verbose_color_func[:,1:]])
        color_tf = np.zeros((3,self.tf_res))
        color_tf[0,:] = 2.0*((lab_color_tf[:,0]-self.min_lab_l) / (self.max_lab_l-self.min_lab_l)) - 1.0
        color_tf[1,:] = 2.0*((lab_color_tf[:,1]-self.min_lab_a) / (self.max_lab_a-self.min_lab_a)) - 1.0
        color_tf[2,:] = 2.0*((lab_color_tf[:,2]-self.min_lab_b) / (self.max_lab_b-self.min_lab_b)) - 1.0

        if get_op_img:
            # and the opacity image
            pil_rgba_img = Image.open(self.imgs_dir+self.img_files[index]).convert("RGBA")
            np_rgba_img = np.array(pil_rgba_img).astype(np.uint8)
            np_opacity_img = np.zeros((np_rgba_img.shape[0],np_rgba_img.shape[1],3)).astype(np.uint8)
            np_opacity_img[:,:,0] = np_rgba_img[:,:,3]
            np_opacity_img[:,:,1] = np_rgba_img[:,:,3]
            np_opacity_img[:,:,2] = np_rgba_img[:,:,3]
            if self.target_opacity_res != pil_rgba_img.width:
                pil_opacity_img = Image.fromarray(np_opacity_img).resize((self.target_opacity_res,self.target_opacity_res), resample=PIL.Image.BICUBIC)
            else:
                pil_opacity_img = Image.fromarray(np_opacity_img)
            target_opacity_img = np.array(pil_opacity_img,dtype=np.float32)/255.0
            target_opacity_img = np.swapaxes(target_opacity_img[:,:,0:1].T,axis1=1,axis2=2)
            return view_params,op_tf,color_tf,target_opacity_img
        else:
            return view_params,op_tf,color_tf
    #

    def __getitem__(self, index):
        if self.n_samples != -1:
            index = index % self.n_samples
        np.random.seed(seed=int(time.time() + index))
        view_params,op_tf,color_tf = self.grab_vis_params(index)

        # now, get the image - opacity and rgb images - reshape it to target resolution, then numpy it
        pil_rgba_img = Image.open(self.imgs_dir+self.img_files[index]).convert("RGBA")
        np_rgba_img = np.array(pil_rgba_img).astype(np.uint8)
        np_rgb_img = np_rgba_img[:,:,0:3]
        np_opacity_img = np.zeros((np_rgba_img.shape[0],np_rgba_img.shape[1],3)).astype(np.uint8)
        np_opacity_img[:,:,0] = np_rgba_img[:,:,3]
        np_opacity_img[:,:,1] = np_rgba_img[:,:,3]
        np_opacity_img[:,:,2] = np_rgba_img[:,:,3]
        if self.target_opacity_res != pil_rgba_img.width:
            if self.jitter > 0:
                rand_jit = np.random.randint(0,self.jitter+1,size=4)
                random_crop = Image.fromarray(np_opacity_img).crop((rand_jit[0],rand_jit[1],np_opacity_img.shape[0]-rand_jit[2],np_opacity_img.shape[1]-rand_jit[3]))
                pil_opacity_img = random_crop.resize((self.target_opacity_res,self.target_opacity_res), resample=PIL.Image.BICUBIC)
            else:
                pil_opacity_img = Image.fromarray(np_opacity_img).resize((self.target_opacity_res,self.target_opacity_res), resample=PIL.Image.BICUBIC)
        else:
            pil_opacity_img = Image.fromarray(np_opacity_img)
        if self.target_rgb_res != pil_rgba_img.width:
            pil_rgb_img = Image.fromarray(np_rgb_img).resize((self.target_rgb_res,self.target_rgb_res), resample=PIL.Image.BICUBIC)
        else:
            pil_rgb_img = Image.fromarray(np_rgb_img).copy()
        '''
        subsampled_pil_img = Image.fromarray(np_rgb_img).resize((self.target_rgb_res//2,self.target_rgb_res//2), resample=PIL.Image.BICUBIC)
        pil_rgb_img = subsampled_pil_img.resize((self.target_rgb_res//1,self.target_rgb_res//1), resample=PIL.Image.BICUBIC)
        '''
        target_opacity_img = np.array(pil_opacity_img,dtype=np.float32)/255.0
        target_opacity_img = np.swapaxes(target_opacity_img[:,:,0:1].T,axis1=1,axis2=2)
        target_rgb_img = np.array(pil_rgb_img,dtype=np.float32)/255.0
        target_rgb_img = np.swapaxes(target_rgb_img.T,axis1=1,axis2=2)

        target_opacity_img = 2.0*target_opacity_img - 1.0
        target_rgb_img = 2.0*target_rgb_img - 1.0

        return torch.from_numpy(view_params),torch.from_numpy(op_tf),torch.from_numpy(color_tf),torch.from_numpy(target_opacity_img),torch.from_numpy(target_rgb_img)
    #
#
