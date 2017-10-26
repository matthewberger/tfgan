import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import net_utils
import numpy as np
import time
from bh_tsne.bhtsne import bh_tsne
from PIL import Image

file_dir = os.path.dirname(os.path.realpath(__file__))
data_generator_dir = os.path.abspath(os.path.join(file_dir, os.pardir)) + '/data_generator'
gan_dir = os.path.abspath(os.path.join(file_dir, os.pardir)) + '/gan'
sys.path.insert(0, data_generator_dir)
sys.path.insert(0, gan_dir)

from model.rendernet import OpacityNetGenerator, TranslationNetGenerator
import tf_generator

class GenerativeVolumeRenderer:
    def __init__(self, opnet, translatenet, scalar_range=None, use_cuda=True, gid=0):
        self.using_cuda = use_cuda

        self.opnet = opnet
        self.translatenet = translatenet
        self.cached_min_bb = None
        self.cached_max_bb = None

        self.zero_color_view = False

        if scalar_range is None:
            self.min_scalar_value = 0
            self.max_scalar_value = 255
        else:
            self.min_scalar_value = scalar_range[0]
            self.max_scalar_value = scalar_range[1]
        self.scalar_range = self.max_scalar_value-self.min_scalar_value
        self.scalar_vals = np.linspace(self.min_scalar_value,self.max_scalar_value,num=256)

        # camera params
        self.elevation = 85.0
        self.azimuth = 330.0
        self.roll = 0.0
        self.zoom = 1.6

        # camera interaction params
        self.min_zoom = 1.0
        self.max_zoom = 2.5
        self.min_elevation = 15
        self.max_elevation = 165

        self.azimuth_delta = 1.0
        self.elevation_delta = 1.0
        self.zoom_delta = (self.max_zoom - self.min_zoom) / 100.0

        # opacity TF params
        self.min_bandwidth = self.scalar_range * 0.05
        self.max_bandwidth = self.scalar_range * 0.4
        self.bandwidth_delta = self.scalar_range * 0.01
        self.gaussian_range = 6*self.min_bandwidth
        subspace_feats = None

        self.norm_eps = 0.0

        self.opnet.eval()
        self.translatenet.eval()
        if self.using_cuda:
            torch.cuda.set_device(gid)
            self.opnet.cuda()
            self.translatenet.cuda()

        # opacity_gmm,color_gmm = tf_generator.generate_opacity_color_gmm(min_scalar_value,max_scalar_value,num_modes,begin_alpha=0.25,end_alpha=0.8)

        use_cached = False
        if use_cached:
            self.opacity_gmm, self.color_gmm = np.load('cached_opacity.npy'), np.load('cached_color.npy')
            #_, self.color_gmm = tf_generator.generate_opacity_color_gmm(self.min_scalar_value, self.max_scalar_value, 1)
        else:
            self.opacity_gmm, self.color_gmm = tf_generator.generate_opacity_color_gmm(self.min_scalar_value, self.max_scalar_value, 2)
        # self.opacity_gmm,self.color_gmm = tf_generator.generate_opacity_color_gmm(self.min_scalar_value,self.max_scalar_value,3,begin_alpha=0.25,end_alpha=0.8)
        self.predicted_sensitivity = torch.ones(256)

        # initialize TF encodings
        self.update_gmm_transfer_function()

        self.use_pca = False
        self.use_opacity_sensitivity = True
        self.use_cached_space = False
        self.use_sampled_tfs = False

    def latent_dim(self):
        return self.opnet.latent_dim

    def toggle_zero_view(self):
        self.zero_color_view = not self.zero_color_view

    def update_gmm_transfer_function(self):
        self.opacity_tf, self.color_tf = tf_generator.generate_tf_from_gmm(self.opacity_gmm, self.color_gmm, self.min_scalar_value, self.max_scalar_value, 256, True)
        self.opnet_encode_tf(self.opacity_tf)
        self.translation_encode_opacity_tf(self.opacity_tf)
        self.translation_encode_color_tf(self.color_tf)

    def gather_vars(self, elevation, azimuth, roll, zoom, is_volatile=True, requires_grad=False):
        # normalize data
        normalized_elevation = net_utils.normalize_elevation(elevation)
        normalized_azimuth = net_utils.normalize_azimuth(azimuth)
        normalized_roll = net_utils.normalize_roll(roll)
        #normalized_zoom = net_utils.normalize_zoom(zoom, self.min_zoom, self.max_zoom)
        normalized_zoom = net_utils.normalize_zoom(zoom, 1.0, 2.5)
        view_params = np.reshape(np.array([normalized_elevation, normalized_azimuth[0], normalized_azimuth[1], normalized_roll, normalized_zoom]), ((1, 5))).astype(np.float32)
        normalized_opacity_func = np.reshape(net_utils.normalize_opacity(self.opacity_tf[:, 1:]), ((1, 1, 256))).astype(np.float32)
        normalized_color_func = np.reshape(net_utils.normalize_rgb_to_lab(self.color_tf[:, 1:]), ((1, 3, 256))).astype(np.float32)

        # convert to PyTorch variables
        if self.using_cuda:
            th_view = Variable(torch.from_numpy(view_params).cuda(), volatile=is_volatile)
            th_opacity = Variable(torch.from_numpy(normalized_opacity_func).cuda(), requires_grad=requires_grad)
            th_color = Variable(torch.from_numpy(normalized_color_func).cuda(), volatile=is_volatile)
        else:
            th_view = Variable(torch.from_numpy(view_params), volatile=is_volatile)
            th_opacity = Variable(torch.from_numpy(normalized_opacity_func), requires_grad=requires_grad)
            th_color = Variable(torch.from_numpy(normalized_color_func), volatile=is_volatile)
        return th_view, th_opacity, th_color

    def torch_view(self, elevation, azimuth, roll, zoom, is_volatile=True, requires_grad=False, replicate=1):
        normalized_elevation = net_utils.normalize_elevation(elevation)
        normalized_azimuth = net_utils.normalize_azimuth(azimuth)
        normalized_roll = net_utils.normalize_roll(roll)
        #normalized_zoom = net_utils.normalize_zoom(zoom, self.min_zoom, self.max_zoom)
        normalized_zoom = net_utils.normalize_zoom(zoom, 1.0, 2.5)
        view_params = np.reshape(np.array([normalized_elevation, normalized_azimuth[0], normalized_azimuth[1], normalized_roll, normalized_zoom]), ((1, 5))).astype(np.float32)

        if self.using_cuda:
            th_view_data = torch.from_numpy(view_params).cuda()
        else:
            th_view_data = torch.from_numpy(view_params)
        if replicate > 1:
            th_view_data = th_view_data.expand(replicate, th_view_data.size()[1])

        return Variable(th_view_data, volatile=is_volatile, requires_grad=requires_grad)

    def torch_opacity_tf(self, tf, is_volatile=True, requires_grad=False, replicate=1):
        normalized_opacity_func = np.reshape(net_utils.normalize_opacity(tf[:, 1:]), ((1, 1, 256))).astype(np.float32)

        if self.using_cuda:
            th_opacity = torch.from_numpy(normalized_opacity_func).cuda()
        else:
            th_opacity = torch.from_numpy(normalized_opacity_func)
        if replicate > 1:
            th_opacity = th_opacity.expand(replicate, th_opacity.size()[1], th_opacity.size()[2])

        return Variable(th_opacity, volatile=is_volatile, requires_grad=requires_grad)

    def torch_color_tf(self, tf, is_volatile=True, requires_grad=False, replicate=1):
        lab_tf = net_utils.normalize_rgb_to_lab(tf[:, 1:])
        normalized_color_func = np.zeros((1, 3, 256), dtype=np.float32)
        for i in range(3):
            normalized_color_func[0, i, :] = lab_tf[:, i]

        if self.using_cuda:
            th_color = torch.from_numpy(normalized_color_func).cuda()
        else:
            th_color = torch.from_numpy(normalized_color_func)
        if replicate > 1:
            th_color = th_color.expand(replicate, th_color.size()[1], th_color.size()[2])

        return Variable(th_color, volatile=is_volatile, requires_grad=requires_grad)

    # opacity net encodings
    def opnet_encode_view(self, elevation, azimuth, roll, zoom, is_volatile=True, requires_grad=False, replicate=1):
        th_view = self.torch_view(elevation, azimuth, roll, zoom, is_volatile, requires_grad, replicate)
        self.opnet_view_encoding = self.opnet.view_subnet(th_view)

    def opnet_encode_tf(self, tf, is_volatile=True, requires_grad=False, eps=0.0, pad_ind=5, replicate=1):
        tf_input = self.torch_opacity_tf(tf, is_volatile, requires_grad, replicate)
        if eps > 0:
            for idx in range(256):
                if idx > pad_ind and idx < 256 - pad_ind:
                    tf_input.data[:, 0, idx] += eps
        self.opnet_tf_encoding = self.opnet.encode_opacity(tf_input)
        return tf_input

    # opacity net decoding
    def opnet_predict_image(self, chopoff_tanh=False):
        b_size = self.opnet_tf_encoding.size()[0]
        opacity_out = self.opnet.opacity_latent_decoder(self.opnet_tf_encoding.view(b_size, self.opnet_tf_encoding.size()[1]))
        joint_view_opacity = torch.cat((self.opnet_view_encoding, opacity_out), 1)
        joint_view_opacity_out = self.opnet.merged_subnet(joint_view_opacity)
        img_encoding = joint_view_opacity_out.view(b_size, joint_view_opacity_out.size()[1], 1, 1)

        for ldx in range(len(self.opnet.decoder_subnet)):
            if chopoff_tanh and ldx == len(self.opnet.decoder_subnet)-1:
                break
            else:
                img_encoding = self.opnet.decoder_subnet[ldx](img_encoding)
        return img_encoding

    # translation net encodings
    def translation_encode_view(self, elevation, azimuth, roll, zoom, is_volatile=True, requires_grad=False, replicate=1):
        th_view = self.torch_view(elevation, azimuth, roll, zoom, is_volatile, requires_grad, replicate)
        self.colornet_view_encoding = self.translatenet.view_encoder(th_view)

    def translation_encode_opacity_tf(self, tf, prior_encoding=None, is_volatile=True, requires_grad=False, replicate=1):
        if prior_encoding is not None:
            latent_opacity = prior_encoding
            self.colornet_op_encoding = self.translatenet.opacity_latent_decoder(latent_opacity)
            return prior_encoding
        else:
            op_input = self.torch_opacity_tf(tf, is_volatile, requires_grad, replicate)
            latent_opacity = self.opnet.encode_opacity(op_input)
            self.colornet_op_encoding = self.translatenet.opacity_latent_decoder(latent_opacity)
            return op_input

    def translation_encode_color_tf(self, tf, is_volatile=True, requires_grad=False, replicate=1):
        color_input = self.torch_color_tf(tf, is_volatile, requires_grad, replicate)
        b_size = color_input.size()[0]
        color_code = self.translatenet.color_encoder(color_input).view(b_size, self.translatenet.ntf)
        latent_color_code = self.translatenet.color_latent_encoder(color_code)
        self.colornet_color_encoding = self.translatenet.color_latent_decoder(latent_color_code)

    # translation net decoding
    def translatenet_predict_image(self, op_image, chopoff_tanh=False):
        b_size = self.colornet_op_encoding.size()[0]
        opacity_encoded = self.colornet_op_encoding.view(b_size, self.colornet_op_encoding.size()[1])
        color_encoded = self.colornet_color_encoding.view(b_size, self.colornet_color_encoding.size()[1])

        # feed in image
        image_encoded = op_image
        all_encoded_images = []
        for image_layer in self.translatenet.image_encoder_layers:
            image_encoded = image_layer(image_encoded)
            # print('post opacity image shape:',image_encoded.size())
            all_encoded_images.append(image_encoded)

        # merge the view, opacity, and color TFs
        joint_vis_encoding = torch.cat((self.colornet_view_encoding, opacity_encoded, color_encoded), 1)

        channel_size = self.colornet_view_encoding.size()[1] + opacity_encoded.size()[1] + color_encoded.size()[1]
        tiled_encoding = joint_vis_encoding.view(b_size, channel_size, 1, 1)
        tiled_encoding = tiled_encoding.expand(b_size, channel_size, self.translatenet.min_spatial_res, self.translatenet.min_spatial_res)

        # merge image with tiled vis encoding
        full_code = torch.cat((image_encoded, tiled_encoding), 1)

        # push through bottleneck layers to fuse features
        fused_encoding = full_code
        for layer in self.translatenet.bottleneck_residual_layers:
            # print('layer:',layer,'input size:',fused_encoding.size())
            fused_encoding = layer(fused_encoding)

        # decode the image, adjoining the feature maps from the image encoding
        decoded_image = fused_encoding
        ddx = 0
        for layer, encoded_image in zip(self.translatenet.image_decoder_layers, reversed(all_encoded_images)):
            joint_encoding = decoded_image if ddx == 0 else torch.cat((decoded_image, encoded_image), 1)
            decoded_image = layer(joint_encoding)
            ddx += 1

        # upsample
        for ldx,layer in enumerate(self.translatenet.upsample_layers):
            if chopoff_tanh and ldx == 1:
                for sdx in range(len(layer)):
                    if sdx != len(layer)-1:
                        decoded_image = layer[sdx](decoded_image)
            else:
                decoded_image = layer(decoded_image)

        return decoded_image

    def encode_inputs(self):
        self.opnet_encode_view(self.elevation,self.azimuth,self.roll,self.zoom)
        self.opnet_encode_tf(self.opacity_tf)
        self.translation_encode_view(self.elevation,self.azimuth,self.roll,self.zoom)
        self.translation_encode_opacity_tf(self.opacity_tf, prior_encoding=self.opnet_tf_encoding)
        self.translation_encode_color_tf(self.color_tf)

    def predict_sensitivity(self, elevation, azimuth, roll, zoom, min_bb=None, max_bb=None, return_img=False):
        # opacity network
        self.opnet_encode_view(elevation,azimuth,roll,zoom, is_volatile=False, requires_grad=False)
        th_opacity = self.opnet_encode_tf(self.opacity_tf, is_volatile=False, requires_grad=True, eps=self.norm_eps)
        predicted_opacity = self.opnet_predict_image()

        # translation network
        if self.use_opacity_sensitivity:
            predicted_img = predicted_opacity
        else:
            self.translation_encode_view(elevation,azimuth,roll,zoom, is_volatile=False, requires_grad=False)
            self.translation_encode_opacity_tf(self.color_tf, prior_encoding = self.opnet_tf_encoding, is_volatile=False, requires_grad=False)
            self.translation_encode_color_tf(self.color_tf, is_volatile=False, requires_grad=False)
            predicted_img = self.translatenet_predict_image(predicted_opacity)

        if min_bb is not None and max_bb is not None:
            if self.use_opacity_sensitivity:
                x_range = np.arange(min_bb[0]//4, max_bb[0]//4 + 1)
                y_range = np.arange(min_bb[1]//4, max_bb[1]//4 + 1)
                predicted_img = predicted_opacity[:, :, y_range[0]:y_range[-1], x_range[0]:x_range[-1]]
            else:
                x_range = np.arange(min_bb[0], max_bb[0] + 1)
                y_range = np.arange(min_bb[1], max_bb[1] + 1)
                predicted_img = predicted_img[:, :, y_range[0]:y_range[-1], x_range[0]:x_range[-1]]

        img_norm = torch.norm(predicted_img)
        img_norm.backward()
        self.predicted_sensitivity = self.sensitivity_filter(torch.Tensor.numpy(th_opacity.grad.data.cpu())[0, 0, :])

        if return_img:
            if self.use_opacity_sensitivity:
                self.translation_encode_view(elevation,azimuth,roll,zoom, is_volatile=False, requires_grad=False)
                self.translation_encode_opacity_tf(self.color_tf, prior_encoding = self.opnet_tf_encoding, is_volatile=False, requires_grad=False)
                self.translation_encode_color_tf(self.color_tf, is_volatile=False, requires_grad=False)
                predicted_img = self.translatenet_predict_image(predicted_opacity)
            return 0.5 * (torch.Tensor.numpy(predicted_img.data.cpu())[0, :, :, :] + 1.0)

    def predict_all_sensitivities(self, elevation, azimuth, roll, zoom, num_blocks):
        real_start = time.time()
        b_size = num_blocks * num_blocks
        # opacity network
        self.opnet_encode_view(elevation,azimuth,roll,zoom, is_volatile=False, requires_grad=False, replicate=b_size)
        th_opacity = self.opnet_encode_tf(self.opacity_tf, is_volatile=False, requires_grad=True, eps=self.norm_eps, replicate=b_size)
        predicted_opacity = self.opnet_predict_image(chopoff_tanh=False)

        # translation network
        if not self.use_opacity_sensitivity:
            self.translation_encode_view(elevation,azimuth,roll,zoom, is_volatile=False, requires_grad=False, replicate=b_size)
            self.translation_encode_opacity_tf(self.color_tf, prior_encoding = self.opnet_tf_encoding, is_volatile=False, requires_grad=True)
            self.translation_encode_color_tf(self.color_tf, is_volatile=False, requires_grad=False, replicate=b_size)
            predicted_imgs = self.translatenet_predict_image(predicted_opacity,chopoff_tanh=False)

        if self.use_opacity_sensitivity:
            block_size = 64 // num_blocks
        else:
            img_res = predicted_imgs.size()[-1]
            block_size = img_res // num_blocks

        if self.using_cuda:
            all_norms = Variable(torch.zeros(b_size, 1).cuda())
            all_ones = Variable(torch.ones(b_size,1).cuda())
        else:
            all_norms = Variable(torch.zeros(b_size, 1))
            all_ones = Variable(torch.ones(b_size,1))

        for r in range(num_blocks):
            for c in range(num_blocks):
                xmin = c * block_size
                xmax = (c + 1) * block_size
                ymin = r * block_size
                ymax = (r + 1) * block_size
                idx = r * num_blocks + c
                if self.use_opacity_sensitivity:
                    all_norms[idx, 0] = predicted_opacity[idx, :, ymin:ymax, xmin:xmax].norm()
                else:
                    all_norms[idx, 0] = predicted_imgs[idx, :, ymin:ymax, xmin:xmax].norm()
        all_norms.backward(all_ones)
        op_tf_grads = torch.Tensor.numpy(th_opacity.grad.data.cpu())[:, 0, :]
        filtered_sensitivities = self.sensitivities_filter(op_tf_grads)

        return filtered_sensitivities

    def sensitivity_filter(self, sensitivities):
        n_scalars = sensitivities.shape[0]
        copy = np.zeros(n_scalars)
        for i in range(n_scalars):
            if i == 0:
                copy[i] = (sensitivities[0]+sensitivities[1]) / 2
            else:
                copy[i] = (sensitivities[i] + sensitivities[i - 1]) / 2
        return copy

    def sensitivities_filter(self, sensitivities):
        b_size,  n_scalars = sensitivities.shape
        copy = np.zeros((b_size, n_scalars))
        for i in range(n_scalars):
            if i == 0:
                copy[:, 0] = (sensitivities[:, 0] + sensitivities[:, 1]) / 2
            else:
                copy[:, i] = (sensitivities[:, i] + sensitivities[:, i - 1]) / 2
        return copy

    def predict_img(self, encode_inputs=True):
        real_start = time.time()

        start_time = time.time()
        if encode_inputs:
            self.encode_inputs()
        self.opacity_img = self.opnet_predict_image()
        self.predicted_img = self.translatenet_predict_image(self.opacity_img)

        start_time = time.time()
        np_img = 0.5 * (torch.Tensor.numpy(self.predicted_img.data.cpu())[0, :, :, :] + 1.0)

        return np_img

    def translatenet_predict_images(self, op_imgs, op_feats, chopoff_tanh=False):
        bsize = op_imgs.size()[0]
        self.colornet_op_encoding = self.translatenet.opacity_latent_decoder(op_feats)
        opacity_encoded = self.colornet_op_encoding.expand(bsize, self.colornet_op_encoding.size()[1])
        color_encoded = self.colornet_color_encoding.expand(bsize, self.colornet_color_encoding.size()[1])
        color_view_encoded = self.colornet_view_encoding.expand(bsize, self.colornet_view_encoding.size()[1])

        # feed in image
        image_encoded = op_imgs
        all_encoded_images = []
        for image_layer in self.translatenet.image_encoder_layers:
            image_encoded = image_layer(image_encoded)
            # print('post opacity image shape:',image_encoded.size())
            all_encoded_images.append(image_encoded)

        # merge the view, opacity, and color TFs
        joint_vis_encoding = torch.cat((color_view_encoded, opacity_encoded, color_encoded), 1)

        channel_size = joint_vis_encoding.size()[1]
        tiled_encoding = joint_vis_encoding.view(bsize, channel_size, 1, 1)
        tiled_encoding = tiled_encoding.expand(bsize, channel_size, self.translatenet.min_spatial_res, self.translatenet.min_spatial_res)

        # merge image with tiled vis encoding
        full_code = torch.cat((image_encoded, tiled_encoding), 1)

        # push through bottleneck layers to fuse features
        fused_encoding = full_code
        for layer in self.translatenet.bottleneck_residual_layers:
            # print('layer:',layer,'input size:',fused_encoding.size())
            fused_encoding = layer(fused_encoding)

        # decode the image, adjoining the feature maps from the image encoding
        decoded_image = fused_encoding
        ddx = 0
        for layer, encoded_image in zip(self.translatenet.image_decoder_layers, reversed(all_encoded_images)):
            joint_encoding = decoded_image if ddx == 0 else torch.cat((decoded_image, encoded_image), 1)
            decoded_image = layer(joint_encoding)
            ddx += 1

        # upsample
        for ldx,layer in enumerate(self.translatenet.upsample_layers):
            if chopoff_tanh and ldx == 1:
                for sdx in range(len(layer)):
                    if sdx != len(layer)-1:
                        decoded_image = layer[sdx](decoded_image)
            else:
                decoded_image = layer(decoded_image)

        return decoded_image

    def opnet_predict_images(self, op_feats):
        op_feats_dim = op_feats.size()
        opacity_out = self.opnet.opacity_latent_decoder(op_feats.view(op_feats_dim[0], op_feats_dim[1]))
        # expand the view
        view_size = self.opnet_view_encoding.size()
        opnet_view_encoding = self.opnet_view_encoding.expand(op_feats_dim[0], self.opnet_view_encoding.size()[1])

        joint_view_opacity = torch.cat((opnet_view_encoding, opacity_out), 1)
        joint_view_opacity_out = self.opnet.merged_subnet(joint_view_opacity)
        img_encoding = joint_view_opacity_out.view(op_feats_dim[0], joint_view_opacity_out.size()[1], 1, 1)
        return self.opnet.decoder_subnet(img_encoding)

    def predict_imgs(self, op_feats):
        real_start = time.time()

        start_time = time.time()
        self.opacity_imgs = self.opnet_predict_images(op_feats)
        self.predicted_imgs = self.translatenet_predict_images(self.opacity_imgs, op_feats)

        start_time = time.time()
        np_imgs = 0.5 * (torch.Tensor.numpy(self.predicted_imgs.data.cpu()) + 1.0)

        return np_imgs

    def save_image(self, name):
        cur_img = self.predict_img()
        np_img = np.zeros((cur_img.shape[1], cur_img.shape[2], 3))
        for i in range(3):
            np_img[:, :, i] = cur_img[i, :, :]
        if self.cached_min_bb is not None:
            x_range = np.arange(self.cached_min_bb[0], self.cached_max_bb[0] + 1)
            y_range = np.arange(self.cached_min_bb[1], self.cached_max_bb[1] + 1)
            for y in y_range:
                np_img[y, x_range, :] += 0.2
            np_img = np.minimum(np_img, 1.0)

        pil_img = Image.fromarray(np.array(255 * np_img, dtype=np.uint8))
        pil_img.save(name)

    def convert_to_bitmap(self, np_img, target_res, min_bb=None, max_bb=None):
        # proper image
        reshaped_img = np.zeros((np_img.shape[1], np_img.shape[2], 3))
        for i in range(3):
            reshaped_img[:, :, i] = np_img[i, :, :]
        if min_bb is not None:
            x_range = np.arange(min_bb[0], max_bb[0] + 1)
            y_range = np.arange(min_bb[1], max_bb[1] + 1)
            for y in y_range:
                reshaped_img[y, x_range, :] += 0.2
            reshaped_img = np.minimum(reshaped_img, 1.0)

        self.cached_min_bb = min_bb
        self.cached_max_bb = max_bb

        if target_res != reshaped_img.shape[0]:
            pil_img = Image.fromarray(np.array(255 * reshaped_img, dtype=np.uint8))
            reshaped_pil_img = pil_img.resize(size=(target_res, target_res), resample=Image.BILINEAR)
            reshaped_np_img = np.array(reshaped_pil_img).astype(np.uint32)
        else:
            reshaped_np_img = (255 * reshaped_img).astype(np.uint32)
        bit_np_img = np.bitwise_or(reshaped_np_img[:, :, 2], np.left_shift(reshaped_np_img[:, :, 1], 8))
        bit_np_img = np.bitwise_or(bit_np_img, np.left_shift(reshaped_np_img[:, :, 0], 16))
        return bit_np_img

    def decode_tf(self, op_feat):
        if self.using_cuda:
            th_op_feat = Variable(torch.FloatTensor(1, self.opnet.latent_dim).cuda(), volatile=True)
        else:
            th_op_feat = Variable(torch.FloatTensor(1, self.opnet.latent_dim), volatile=True)
        th_op_feat.data.copy_(op_feat.data.view(1,self.opnet.latent_dim))
        reconstructed_tf = self.opnet.opacity_latent_reconstructor(th_op_feat)
        return 0.5*(torch.Tensor.numpy(reconstructed_tf.data[0,:].cpu())+1.0)

    def compute_latent_space_bounds(self, n_samples=250, begin_alpha=0.05, end_alpha=0.95):
        op_gmms = []
        for _ in range(n_samples):
            op_gmm, _ = tf_generator.generate_opacity_color_gmm(self.min_scalar_value, self.max_scalar_value, np.random.randint(1, 5), begin_alpha=begin_alpha, end_alpha=end_alpha)
            #op_gmm, _ = tf_generator.generate_opacity_color_gmm(self.min_scalar_value, self.max_scalar_value, 1, begin_alpha=begin_alpha, end_alpha=end_alpha)
            op_gmms.append(op_gmm)
        random_op_tfs = []
        for op_gmm in op_gmms:
            op_tf = tf_generator.generate_tf_from_gmm(op_gmm, None, self.min_scalar_value, self.max_scalar_value, 256, True)
            random_op_tfs.append(op_tf)
        end = time.time()
        random_op_tfs = np.array(random_op_tfs)

        random_op_feats = []
        for random_op_tf in random_op_tfs:
            normalized_opacity_func = np.reshape(net_utils.normalize_opacity(random_op_tf[:, 1:]), ((1, 1, 256))).astype(np.float32)
            if self.using_cuda:
                th_opacity = Variable(torch.from_numpy(normalized_opacity_func).cuda(), volatile=True)
            else:
                th_opacity = Variable(torch.from_numpy(normalized_opacity_func))
            latent_opacity = self.opnet.encode_opacity(th_opacity)
            op_feat = torch.Tensor.numpy(latent_opacity.data.cpu())[0, :]
            random_op_feats.append(op_feat)
        end = time.time()
        random_op_feats = np.array(random_op_feats)
        return np.min(random_op_feats, axis=0), np.max(random_op_feats, axis=0), random_op_feats

    def generate_tf_space(self, n_boundary_samples=1000, n_latent_samples=10000, mb_size=100):
        op_feats_tmp_filename = '/tmp/op_feats.npy'
        projected_feats_tmp_filename = '/tmp/projected_feats.npy'
        op_tfs_tmp_filename = "/tmp/op_tfs.npy"

        if self.use_cached_space and os.path.exists(op_feats_tmp_filename) and os.path.exists(projected_feats_tmp_filename) and os.path.exists(op_tfs_tmp_filename):
            random_op_feats = np.load(op_feats_tmp_filename)
            random_op_tfs = np.load(op_tfs_tmp_filename)
            projected_op_feats = np.load(projected_feats_tmp_filename)
            return random_op_feats, projected_op_feats, random_op_tfs

        # compute latent space bounds - sample from latent space, uniformly at random
        min_op_feats,max_op_feats,sampled_op_feats = self.compute_latent_space_bounds(n_boundary_samples)
        print('min op feats:',min_op_feats,'max op feats:',max_op_feats)
        latent_sampling = np.random.rand(n_latent_samples,self.opnet.latent_dim)
        latent_sampling = min_op_feats*(1.0-latent_sampling) + max_op_feats*latent_sampling

        '''
        mean_feats = (max_op_feats+min_op_feats)/2
        std_feats = (max_op_feats-min_op_feats)/2
        latent_sampling = std_feats*np.random.randn(n_latent_samples,self.opnet.latent_dim) + mean_feats
        '''
        latent_sampling = latent_sampling.astype(np.float32)

        write_some_tfs = False
        write_all_tfs = True

        ae_tfs = []
        ae_op_feats = []
        n_batches = latent_sampling.shape[0]//mb_size
        if self.using_cuda:
            th_latent_samples = Variable(torch.FloatTensor(mb_size, self.opnet.latent_dim).cuda(), volatile=True)
        else:
            th_latent_samples = Variable(torch.FloatTensor(mb_size, self.opnet.latent_dim), volatile=True)
        start = time.time()
        for bdx in range(n_batches):
            latent_batch = torch.from_numpy(latent_sampling[bdx*mb_size:(bdx+1)*mb_size,:])
            th_latent_samples.data.copy_(latent_batch)
            reconstructed_tfs = self.opnet.opacity_latent_reconstructor(th_latent_samples)
            recon_tfs = 0.5*(torch.Tensor.numpy(reconstructed_tfs.data.cpu())+1.0)
            if bdx==0 and write_some_tfs:
                for idx in range(recon_tfs.shape[0]):
                    proper_tf = np.vstack((self.scalar_vals,recon_tfs[idx,:]))
                    np.save('sampled_tf_'+str(idx)+'.npy',proper_tf.T)
            reconstructed_tfs = reconstructed_tfs.view(mb_size,1,256)
            ae_tfs.extend([recon_tfs[idx,:] for idx in range(mb_size)])
            latent_opacity = self.opnet.encode_opacity(reconstructed_tfs)
            op_feats = torch.Tensor.numpy(latent_opacity.data.cpu())
            ae_op_feats.extend([op_feats[idx] for idx in range(mb_size)])
        ae_tfs = np.array(ae_tfs)

        self.ae_op_feats = np.array(ae_op_feats)
        if self.use_sampled_tfs:
            self.ae_op_feats = sampled_op_feats
        U, s, V = np.linalg.svd(self.ae_op_feats, full_matrices=False)
        print('ae singular values:', s)

        self.tf_subspace = np.zeros((self.opnet.latent_dim, 2))
        self.tf_subspace[:, 0] = V[0, :].T
        self.tf_subspace[:, 1] = V[1, :].T
        self.update_subspace_bounds()

        reconstructed_feats = (self.tf_subspace.dot(self.tf_subspace.T.dot(self.ae_op_feats.T))).T
        relative_recon_err = np.linalg.norm(reconstructed_feats - self.ae_op_feats) / np.linalg.norm(self.ae_op_feats)
        random_op_feats = self.subspace_feats.dot(self.tf_subspace.T)

        if not self.use_pca:
            start = time.time()
            random_op_feats = self.ae_op_feats
            self.subspace_feats = np.array([projected_pt for projected_pt in bh_tsne(self.ae_op_feats, no_dims=2, perplexity=30, theta=0.5, randseed=-1, verbose=True)])
            tsne_U, tsne_s, tsne_V = np.linalg.svd(self.subspace_feats, full_matrices=False)
            self.subspace_feats = self.subspace_feats.dot(tsne_V.T)
            end = time.time()

        save_feats = True
        if save_feats:
            np.save(op_feats_tmp_filename, random_op_feats)
            np.save(projected_feats_tmp_filename, self.subspace_feats)
            np.save(op_tfs_tmp_filename, ae_tfs)

        return random_op_feats, self.subspace_feats, ae_tfs

    def update_subspace_bounds(self):
        self.subspace_feats = self.ae_op_feats.dot(self.tf_subspace)
        self.projection_errors = np.linalg.norm(self.ae_op_feats-self.subspace_feats.dot(self.tf_subspace.T), axis=1)
        self.subspace_ll = np.min(self.subspace_feats, axis=0).reshape((2, 1))
        self.subspace_ur = np.max(self.subspace_feats, axis=0).reshape((2, 1))

    def nearest_projection(self, pc, dim):
        # first: find nearest vector for pc with constrained dim on l2 ball
        constrained_val = self.tf_subspace[dim,pc]
        if np.abs(constrained_val) == 1:
            self.tf_subspace[:,pc] = 0
            self.tf_subspace[dim,pc] = constrained_val
        else:
            prev_vec = np.array(self.tf_subspace[:,pc])
            for idx in range(200):
                self.tf_subspace[:,pc] /= np.linalg.norm(self.tf_subspace[:,pc])
                self.tf_subspace[dim,pc] = constrained_val
                prev_vec = np.array(self.tf_subspace[:,pc])

        # second: Gram-Schmidt to find other vector
        other_pc = int(1-pc)
        self.tf_subspace[:,other_pc] -= (self.tf_subspace[:,pc].dot(self.tf_subspace[:,other_pc]))*self.tf_subspace[:,pc]

    def sample_subspace(self, normalized_pt):
        n_pt = normalized_pt.reshape((2, 1))
        projected_pt = self.subspace_ll * (1.0 - n_pt) + self.subspace_ur * n_pt
        #print('n pt shape:', n_pt.shape, 'projected pt shape:', self.subspace_ll.shape, 'subspace shape:', self.tf_subspace.shape)
        return self.tf_subspace.dot(projected_pt)
