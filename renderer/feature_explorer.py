import sys
import os

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QColorDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtSvg import QSvgGenerator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib
from scipy import spatial
from genren import GenerativeVolumeRenderer

import torch
from torch.autograd import Variable

import numpy as np
from PIL import Image
import argparse

file_dir = os.path.dirname(os.path.realpath(__file__))
data_generator_dir = os.path.abspath(os.path.join(file_dir, os.pardir)) + '/data_generator'
sys.path.insert(0, data_generator_dir)

import tf_generator

matplotlib.rcParams.update({'font.size': 10})
#matplotlib.rcParams['font.weight'] = 'heavy'
matplotlib.rcParams['font.family'] = 'Times New Roman'

class TFExplorer(QWidget):
    def __init__(self, opt):
        super().__init__()

        self.img_size = 256
        self.block_size = 192
        self.tf_height = 200
        self.block_dim = 4
        self.color_cm_height = 20

        self.img_view_size = self.block_dim*self.block_size
        self.width = self.img_size + self.img_view_size
        self.height = self.img_view_size

        self.use_cuda = opt.cuda
        self.border_color = '#43a2ca'

        if self.use_cuda:
            tr = torch.load(opt.trnet, map_location={'cuda:0': 'cuda:%d' % opt.gid, 'cuda:1': 'cuda:%d' % opt.gid})
        else:
            tr = torch.load(opt.trnet, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu'})
        op = tr.opNet[0]

        if opt.range is None:
            scalar_range = None
        else:
            scalar_range = np.load(opt.range)

        self.genren = GenerativeVolumeRenderer(op, tr, scalar_range=scalar_range, use_cuda=opt.cuda, gid=opt.gid)

        # setup initial data
        self.np_imgs = np.zeros([self.block_dim * self.block_dim, self.block_size, self.block_size, 3]).astype(np.uint32)

        # show the window
        self.setWindowTitle('TF Exploration')
        self.initUI()
        self.layout_window()
        self.activateWindow()
        self.setFocus()
        self.do_update()

    def layout_window(self):
        # top right volume rendering image
        self.image_viewer = ImageViewer(self, self.genren, self.img_size)
        self.image_viewer.setGeometry(self.img_view_size, 0, self.img_size, self.img_size)
        self.image_viewer.show()

        # bottom right feature space scatterplot
        self.scatterplot = TFFeatureSpace(self.genren, self)
        self.scatterplot.setGeometry(self.img_view_size, self.img_size+self.tf_height+self.color_cm_height, self.img_size, self.img_size)
        self.scatterplot.show()
        self.genren.opnet_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        self.genren.translation_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)

        # decoded transfer function
        self.decoded_tf = DecodedTF(self.genren, self, width=self.img_size, height=self.tf_height)
        self.decoded_tf.setGeometry(self.img_view_size, self.img_size, self.img_size, self.tf_height)
        self.decoded_tf.show()

        # color transfer function
        self.ctf_editor = ColorTFEditor(self.genren, self, width=self.img_size, height=self.color_cm_height)
        self.ctf_editor.setGeometry(self.img_view_size, self.img_size+self.tf_height, self.img_size, self.color_cm_height)
        self.ctf_editor.show()

        # main viewer
        self.pixmap = QPixmap(self.img_view_size, self.img_view_size)
        self.q_img = None
        self.image_label = QLabel(self)
        self.image_label.setGeometry(0, 0, self.img_view_size, self.img_view_size)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.show()

    def initUI(self):
        d_geo = QApplication.desktop().screenGeometry()
        d_width = d_geo.width()
        d_height = d_geo.height()
        self.setGeometry(d_width / 2 - self.width / 2,d_height / 2 - self.height / 2,self.width, self.height)
        self.setFixedSize(self.width, self.height)
        self.show()

    def update_image_viewer(self):
        self.image_viewer.do_update()
        self.decoded_tf.plot_update()

    def post_process_image(self,img,left_pad=4, right_pad=4, bottom_pad=4, top_pad=4):
        hex_color = self.border_color[1:]
        border_colors = tuple(int(hex_color[i:i+2], 16) for i in (0, 2 ,4))
        for idx in range(3):
            remapped_color = border_colors[idx]/255.0
            img[idx,:,:left_pad] = remapped_color
            img[idx,:,-right_pad:] = remapped_color
            img[idx,:top_pad,:] = remapped_color
            img[idx,-bottom_pad:,:] = remapped_color

    def update_blocks(self):
        self.scatterplot.update_np_imgs()
        tiled_bit_img = np.zeros((self.img_view_size,self.img_view_size),dtype=np.int32)
        bd = self.block_size
        for r in range(self.block_dim):
            for c in range(self.block_dim):
                idx = r * self.block_dim + c
                left_pad,right_pad,bottom_pad,top_pad = 4,4,4,4
                if c == 0:
                    left_pad *= 2
                if c == self.block_dim-1:
                    right_pad *= 2
                if r == 0:
                    top_pad *= 2
                if r == self.block_dim-1:
                    bottom_pad *= 2
                self.post_process_image(self.np_imgs[idx], left_pad,right_pad,bottom_pad,top_pad)
                bit_img = self.genren.convert_to_bitmap(self.np_imgs[idx], self.block_size)
                tiled_bit_img[r*bd:(r+1)*bd,c*bd:(c+1)*bd] = self.genren.convert_to_bitmap(self.np_imgs[idx], bd)

        self.q_img = QImage(tiled_bit_img.flatten(), self.img_view_size, self.img_view_size, QImage.Format_RGB32)
        self.pixmap.convertFromImage(self.q_img)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.repaint()

    def do_update(self):
        self.update_image_viewer()
        self.update_blocks()

    def keyPressEvent(self, event):
        if (event.key() == QtCore.Qt.Key_Escape):
            sys.exit(0)
        if event.key() == QtCore.Qt.Key_Shift:
            self.scatterplot.shift_mod = True
            self.scatterplot.update_plot()
        if event.key() == QtCore.Qt.Key_R:
            self.scatterplot.reset_view_range()
            self.scatterplot.update_plot()
            #self.do_update()
        if event.key() == QtCore.Qt.Key_Q:
            _, self.genren.color_gmm = tf_generator.generate_opacity_color_gmm(self.genren.min_scalar_value, self.genren.max_scalar_value, 1)
            self.genren.update_gmm_transfer_function()
            self.do_update()
            self.ctf_editor.update_plot()
        if event.key() == QtCore.Qt.Key_Z:
            self.scatterplot.update_view_range()
            self.scatterplot.update_plot()
        if event.key() == QtCore.Qt.Key_C:
            np.save('cached_opacity.npy',self.genren.opacity_gmm)
            np.save('cached_color.npy',self.genren.color_gmm)
        if event.key() == QtCore.Qt.Key_P:
            self.genren.use_pca = not self.genren.use_pca
        if event.key() == QtCore.Qt.Key_S:
            def generate_unique_filename(name, ext):
                found_filename = False
                fdx = 0
                unique_filename = name+'-'+str(fdx)+'.'+ext
                while not found_filename:
                    if os.path.isfile(unique_filename):
                        fdx+=1
                        unique_filename = name+'-'+str(fdx)+'.'+ext
                    else:
                        break
                return unique_filename
            #

            self.image_viewer.save_image(generate_unique_filename('focus_image','png'))
            self.save_image(generate_unique_filename('tiled_image','png'))
            self.scatterplot.save_fig(generate_unique_filename('scatterplot', 'png'))
            self.decoded_tf.save_fig(generate_unique_filename('decoded_tf', 'svg'))
            self.ctf_editor.save_fig(generate_unique_filename('ctf_editor', 'svg'))

    def save_image(self, filename):
        self.scatterplot.update_np_imgs()
        tiled_img = np.zeros((3,self.block_dim*256,self.block_dim*256),dtype=np.float32)
        bd = 256
        for r in range(self.block_dim):
            for c in range(self.block_dim):
                idx = r * self.block_dim + c
                left_pad,right_pad,bottom_pad,top_pad = 4,4,4,4
                if c == 0:
                    left_pad *= 2
                if c == self.block_dim-1:
                    right_pad *= 2
                if r == 0:
                    top_pad *= 2
                if r == self.block_dim-1:
                    bottom_pad *= 2
                self.post_process_image(self.np_imgs[idx], left_pad,right_pad,bottom_pad,top_pad)
                tiled_img[:,r*bd:(r+1)*bd,c*bd:(c+1)*bd] = self.np_imgs[idx]

        tiled_img *= 255
        tiled_img = tiled_img.astype(np.uint8)
        tiled_rgb_img = np.zeros((tiled_img.shape[1],tiled_img.shape[2],3),dtype=np.uint8)
        for i in range(3):
            tiled_rgb_img[:,:,i] = tiled_img[i,:,:]
        tiled_pil_img = Image.fromarray(tiled_rgb_img)
        tiled_pil_img.save(filename)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Shift:
            self.scatterplot.shift_mod = False
            self.scatterplot.update_plot()


class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=256, height=256, keep_frame=True, dpi=100):
        if keep_frame:
            self.fig, self.axis = plt.subplots(figsize=(width, height))
        else:
            self.fig = plt.figure(frameon=False)
            self.axis = self.fig.add_axes([0, 0, 1, 1])
            self.fig.set_figwidth(width)
            self.fig.set_figheight(height)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)
        self.compute_initial_figure()

    def compute_initial_figure(self):
        pass

    def save_fig(self, fig_filename):
        self.fig.savefig(fig_filename)


class TFFeatureSpace(MyMplCanvas):
    def __init__(self, genren, *args, **kwargs):
        self.genren = genren
        self.main_interface = args[0]
        self.selection_thresh = 0.05
        self.tf_dot_size = 8
        self.block_dim = self.main_interface.block_dim
        self.block_size = self.main_interface.block_size
        MyMplCanvas.__init__(self, *args, **kwargs, keep_frame=False)
        self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.shift_mod = False
        self.mouse_pressed = False
        self.tf_scatter = None
        self.zoom = False
        self.dist_cm = matplotlib.cm.get_cmap(name='Reds')
        self.selector = RectangleSelector(self.axis, self.select_callback,drawtype='box', useblit=False,button=[1, 3],interactive=True)

    def form_gaussian_cm(self):
        normalized_vals = np.linspace(0,1,num=50)
        cdict = dict()
        cdict['red'] = [(x, 64/255., 64/255.) for x in normalized_vals]
        cdict['green'] = [(x, 64/255., 64/255.) for x in normalized_vals]
        cdict['blue'] = [(x, 127/255., 127/255.) for x in normalized_vals]
        #cdict['alpha'] = [(x, a, a) for x,a in zip(normalized_vals,alphas)]
        cdict['alpha'] = [(x, x, x) for x in normalized_vals]
        return LinearSegmentedColormap('gaussian_cm', cdict)

    def compute_gaussian_blur(self):
        x_coords,y_coords = np.meshgrid(np.linspace(self.xrange[0],self.xrange[1],self.grid_res), np.linspace(self.yrange[0],self.yrange[1],self.grid_res))
        selected_x_img = self.selected_x*np.ones((self.grid_res,self.grid_res))
        selected_y_img = self.selected_y*np.ones((self.grid_res,self.grid_res))
        sqd_dist_img = (x_coords-selected_x_img)**2 + (y_coords-selected_y_img)**2
        gaussian_img = np.exp(-sqd_dist_img / (self.shephard_bandwidth**2))
        gaussian_img = np.flip(gaussian_img,0)
        return gaussian_img

    def update_np_imgs(self):
        num_blocks = self.block_dim
        xmin = self.zx[0]
        xinc = (self.zx[1] - self.zx[0]) / num_blocks
        ymax = self.zy[1]
        yinc = (self.zy[1] - self.zy[0]) / num_blocks
        op_feats = np.zeros([num_blocks * num_blocks, self.genren.latent_dim()]).astype(np.float32)
        has_feat = np.zeros([num_blocks * num_blocks]).astype(np.bool)
        for j in range(num_blocks):
            for i in range(num_blocks):
                x = xmin + xinc/2 + i*xinc
                y = ymax + -(yinc/2 + j*yinc)
                radius = np.max([xinc,yinc])
                pt = np.array([x,y])

                grid_points = self.compute_neighbor_ball(pt, radius)
                feat, is_feat = self.grid_interpolation_feature(pt, radius/3, grid_points)
                #feat, is_feat = self.shephard_interpolation_feature([x, y])
                if is_feat:
                    has_feat[j * num_blocks + i] = True
                    reshaped_feat = np.reshape(feat, (1, self.genren.latent_dim())).astype(np.float32)
                    op_feats[j * num_blocks + i, :] = reshaped_feat
        if self.genren.using_cuda:
            feats = Variable(torch.from_numpy(op_feats).cuda(), volatile=True)
        else:
            feats = Variable(torch.from_numpy(op_feats))

        np_imgs = self.genren.predict_imgs(feats)
        for i in range(num_blocks * num_blocks):
            if not has_feat[i]:
                np_imgs[i] = np.zeros(np_imgs.shape[1:])
        self.main_interface.np_imgs = np_imgs

    def select_callback(self, eclick, erelease):
        self.starting = [eclick.xdata, eclick.ydata]
        self.ending = [erelease.xdata, erelease.ydata]
        bounds = self.get_bounds()
        self.zx = [bounds[0][0], bounds[1][0]]
        self.zy = [bounds[0][1], bounds[1][1]]
        self.update_plot()
        self.main_interface.update_blocks()

    def compute_initial_figure(self):
        self.tf_feats, self.tf_projection, self.op_tfs = self.genren.generate_tf_space()
        x_data = self.tf_projection[:, 0]
        y_data = self.tf_projection[:, 1]
        self.kd_tree = spatial.KDTree(self.tf_projection)
        min_tf_proj = np.min(self.tf_projection, axis=0)
        max_tf_proj = np.max(self.tf_projection, axis=0)
        self.projection_diag = np.linalg.norm(min_tf_proj - max_tf_proj)
        self.shephard_bandwidth = (self.projection_diag * self.selection_thresh / 2)
        print('shephard bandwidth:',self.shephard_bandwidth)

        # set default rendering data
        reshaped_tf_feat = np.reshape(self.tf_feats[0], ((1, self.genren.latent_dim()))).astype(np.float32)
        if self.genren.using_cuda:
            feat = Variable(torch.from_numpy(reshaped_tf_feat).cuda(), volatile=True)
        else:
            feat = Variable(torch.from_numpy(reshaped_tf_feat))
        self.main_interface.image_viewer.cur_feat = feat

        self.xrange = np.array([np.min(x_data), np.max(x_data)])
        self.yrange = np.array([np.min(y_data), np.max(y_data)])
        self.axis.set_axis_off()
        self.axis.autoscale(False)
        self.axis.set_xlim(self.xrange)
        self.axis.set_ylim(self.yrange)

        # current display range
        self.ll = [self.xrange[0], self.yrange[0]]
        self.ur = [self.xrange[1], self.yrange[1]]
        self.zx = self.xrange
        self.zy = self.yrange

        # --- boundary stuff --- #
        self.line_color = self.main_interface.border_color

        self.border_plots = []
        self.border_plots.append(self.axis.plot([self.zx[0],self.zx[0]], [self.zy[0],self.zy[1]], color=self.line_color, linewidth=2)[0])
        self.border_plots.append(self.axis.plot([self.zx[1],self.zx[1]], [self.zy[0],self.zy[1]], color=self.line_color, linewidth=2)[0])
        self.border_plots.append(self.axis.plot([self.zx[0],self.zx[1]], [self.zy[0],self.zy[0]], color=self.line_color, linewidth=2)[0])
        self.border_plots.append(self.axis.plot([self.zx[0],self.zx[1]], [self.zy[1],self.zy[1]], color=self.line_color, linewidth=2)[0])

        self.sub_border_plots = []
        for y in range(self.main_interface.block_dim-1):
            y_alpha = float(y+1)/self.main_interface.block_dim
            y_pos = (1-y_alpha)*self.zy[0]+y_alpha*self.zy[1]
            for x in range(self.main_interface.block_dim-1):
                x_alpha = float(x+1)/self.main_interface.block_dim
                x_pos = (1-x_alpha)*self.zx[0]+x_alpha*self.zx[1]
                self.sub_border_plots.append(self.axis.plot([self.zx[0],self.zx[1]], [y_pos,y_pos], color=self.line_color, linewidth=2)[0])
                self.sub_border_plots.append(self.axis.plot([x_pos,x_pos], [self.zy[0],self.zy[1]], color=self.line_color, linewidth=2)[0])

        self.selected_x = np.mean(self.xrange)
        self.selected_y = np.mean(self.yrange)

        # tSNE projection
        self.axis.set_axis_off()
        self.axis.autoscale(False)
        self.scatterplot = self.axis.scatter(self.tf_projection[:, 0], self.tf_projection[:, 1], s=self.tf_dot_size, c='#60B26E', alpha=0.1, zorder=1)

        # blur surrounding selection
        self.grid_res = 256
        self.blur_img = self.axis.imshow(self.compute_gaussian_blur(), cmap=self.form_gaussian_cm(), interpolation='bicubic', extent=(self.xrange[0],self.xrange[1],self.yrange[0],self.yrange[1]), aspect='auto', zorder=2)

        # selection
        self.selection_scatterplot = self.axis.scatter([self.selected_x], [self.selected_y], s=self.tf_dot_size, alpha=1.0, color='#212168', zorder=3)

        self.view_x = self.xrange
        self.view_y = self.yrange
    #

    def compute_distance(self, tf):
        self.distance_colors = []
        for t in self.op_tfs:
            self.distance_colors.append(np.linalg.norm(t - tf))
        self.distance_colors = np.array(self.distance_colors)
        mini = np.min(self.distance_colors)
        maxi = np.max(self.distance_colors)
        self.distance_colors /= maxi - mini

    def update_view_range(self):
        self.view_x = self.zx
        self.view_y = self.zy

    def reset_view_range(self):
        self.view_x = self.xrange
        self.view_y = self.yrange

    def update_plot(self):
        self.axis.set_xlim(self.view_x)
        self.axis.set_ylim(self.view_y)

        # selection border
        self.border_plots[0].set_xdata([self.zx[0],self.zx[0]])
        self.border_plots[0].set_ydata([self.zy[0],self.zy[1]])
        self.border_plots[1].set_xdata([self.zx[1],self.zx[1]])
        self.border_plots[1].set_ydata([self.zy[0],self.zy[1]])
        self.border_plots[2].set_xdata([self.zx[0],self.zx[1]])
        self.border_plots[2].set_ydata([self.zy[0],self.zy[0]])
        self.border_plots[3].set_xdata([self.zx[0],self.zx[1]])
        self.border_plots[3].set_ydata([self.zy[1],self.zy[1]])

        # sub borders
        pdx=0
        for y in range(self.main_interface.block_dim-1):
            y_alpha = float(y+1)/self.main_interface.block_dim
            y_pos = (1-y_alpha)*self.zy[0]+y_alpha*self.zy[1]
            for x in range(self.main_interface.block_dim-1):
                x_alpha = float(x+1)/self.main_interface.block_dim
                x_pos = (1-x_alpha)*self.zx[0]+x_alpha*self.zx[1]
                self.sub_border_plots[pdx].set_xdata([self.zx[0],self.zx[1]])
                self.sub_border_plots[pdx].set_ydata([y_pos,y_pos])
                pdx+=1
                self.sub_border_plots[pdx].set_xdata([x_pos,x_pos])
                self.sub_border_plots[pdx].set_ydata([self.zy[0],self.zy[1]])
                pdx+=1
            #
        #

        # selection point
        self.selection_scatterplot.set_offsets(np.array([[self.selected_x],[self.selected_y]]))

        # selection blur
        self.blur_img.set_data(self.compute_gaussian_blur())

        self.draw()

    def onpress(self, event):
        self.mouse_pressed = True
        self.starting = [event.xdata, event.ydata]

    def compute_neighbor_ball(self, point, radius):
        return np.array(self.kd_tree.query_ball_point(point, radius, p=2), dtype=np.int32)

    def shephard_interpolation_feature(self, point):
        if self.genren.use_pca:
            normalized_pt = point
            normalized_pt[0] = (normalized_pt[0]-self.ll[0]) / (self.ur[0]-self.ll[0])
            normalized_pt[1] = (normalized_pt[1]-self.ll[1]) / (self.ur[1]-self.ll[1])
            print('sampling subspace...',normalized_pt)
            return self.genren.sample_subspace(np.array(normalized_pt)),True

        queried_pt_inds = self.kd_tree.query_ball_point(point, self.projection_diag * self.selection_thresh, p=2)
        if (len(queried_pt_inds) == 0):
            return None, False
        pt_weights = np.exp(-(np.linalg.norm((self.tf_projection[queried_pt_inds, :] - point), axis=1) / self.shephard_bandwidth)**2)
        normalized_pt_weights = pt_weights / np.sum(pt_weights)
        return np.sum((self.tf_feats[queried_pt_inds, :].T * normalized_pt_weights).T, axis=0), True

    def grid_interpolation_feature(self, point, bandwidth, grid_points):
        if self.genren.use_pca and True:
            normalized_pt = point
            normalized_pt[0] = (normalized_pt[0]-self.ll[0]) / (self.ur[0]-self.ll[0])
            normalized_pt[1] = (normalized_pt[1]-self.ll[1]) / (self.ur[1]-self.ll[1])
            return self.genren.sample_subspace(np.array(normalized_pt)),True
        if grid_points.shape[0]==0:
            return None, False
        pt_weights = np.exp(-(np.linalg.norm((self.tf_projection[grid_points, :] - point), axis=1) / bandwidth)**2)
        normalized_pt_weights = pt_weights / np.sum(pt_weights)
        return np.sum((self.tf_feats[grid_points, :].T * normalized_pt_weights).T, axis=0), True

    def onmotion(self, event):
        if self.mouse_pressed:
            if event.xdata is not None and event.ydata is not None:
                self.ending = [event.xdata, event.ydata]
            elif event.xdata is not None:
                if self.ydata < self.starting[1]:
                    self.ending = [event.xdata, self.yrange[0]]
                else:
                    self.ending = [event.xdata, self.yrange[0]]
            elif event.ydata is not None:
                if self.xdata < self.starting[0]:
                    self.ending = [self.xrange[0], event.ydata]
                else:
                    self.ending = [self.xrange[1], event.ydata]
            else:
                endingx = self.xrange[0] if self.ending[0] < self.starting[0] else self.xrange[1]
                endingy = self.yrange[0] if self.ending[1] < self.starting[1] else self.yrange[1]
                self.ending = [endingx, endingy]
        elif self.shift_mod:
            plot_pt = np.array([event.xdata, event.ydata])
            fig_size = self.fig.get_size_inches() * self.fig.dpi
            figure_scale = (self.selection_thresh * fig_size[0])**2
            self.selected_x,self.selected_y = event.xdata,event.ydata
            self.update_plot()
            new_tf_feature, has_feature = self.shephard_interpolation_feature(plot_pt)
            if not has_feature:
                return
            reshaped_tf_feat = np.reshape(new_tf_feature, ((1, self.genren.latent_dim()))).astype(np.float32)
            if self.genren.using_cuda:
                feat = Variable(torch.from_numpy(reshaped_tf_feat).cuda(), volatile=True)
            else:
                feat = Variable(torch.from_numpy(reshaped_tf_feat))
            self.main_interface.image_viewer.cur_feat = feat

            self.main_interface.update_image_viewer()

    def get_drawing_distance(self):
        x_dist = np.abs((self.ending[0] - self.starting[0]) / (self.xrange[1] - self.xrange[0]))
        y_dist = np.abs((self.ending[1] - self.starting[1]) / (self.yrange[1] - self.yrange[0]))
        return x_dist, y_dist

    def get_bounds(self):
        lowerleft = [min(self.ending[0], self.starting[0]), min(self.ending[1], self.starting[1])]
        upperright = [max(self.ending[0], self.starting[0]), max(self.ending[1], self.starting[1])]
        return [lowerleft, upperright]

    def onrelease(self, event):
        if self.mouse_pressed:
            self.onmotion(event)
            self.mouse_pressed = False

class DecodedTF(MyMplCanvas):
    def __init__(self, genren, *args, **kwargs):
        self.tf_line_width = 1
        self.genren = genren
        self.main_interface = args[0]
        MyMplCanvas.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        self.min_data = np.min(self.genren.opacity_tf[:, 0])
        self.max_data = np.max(self.genren.opacity_tf[:, 0])
        decoded_tf = self.genren.decode_tf(self.main_interface.image_viewer.cur_feat)
        self.axis.set_ylim([0, 1])
        self.axis.set_title('Decoded Opacity TF')
        self.axis.set_xlim([self.min_data, self.max_data])
        self.tf_plot, = self.axis.plot(self.genren.opacity_tf[:, 0], decoded_tf, linewidth=self.tf_line_width, color='darkblue')
        #self.axis.set_ylabel('Opacity', color='darkblue')
        #self.axis.set_xlabel('Scalar Value', color='darkblue')

    def plot_update(self, do_full_update=False):
        decoded_tf = self.genren.decode_tf(self.main_interface.image_viewer.cur_feat)
        self.tf_plot.set_ydata(decoded_tf)
        self.draw()

class ColorTFEditor(MyMplCanvas):
    def __init__(self, genren, *args, **kwargs):
        self.genren = genren
        self.main_interface = args[0]
        self.tf_dot_size = 16
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.pt_selected = None
        self.pressed_plot_pt = None
        self.modifying_modes = False

        self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.axis.axis('off')

    def compute_initial_figure(self):
        self.update_gmms()
        self.axis.set_xlim([self.genren.min_scalar_value,self.genren.max_scalar_value])
        self.axis.set_ylim([0,1])
        cm_data = np.tile(self.genren.color_tf[:, 1:], (6, 1, 1))
        n_modes = self.genren.color_gmm.shape[0]
        self.cm_params_plot = self.axis.scatter(self.genren.color_gmm[:, 0],[0.5 for _ in range(n_modes)],s=self.tf_dot_size, zorder=2)
        self.cm_plot = self.axis.imshow(cm_data, extent=(self.genren.min_scalar_value,self.genren.max_scalar_value,0,1), aspect='auto', zorder=1)

    def cv_to_cm(self, v):
        """color value to color map value"""
        cv_min = self.genren.color_tf[0][0]
        cv_max = self.genren.color_tf[-1][0]
        cm_min = 0
        cm_max = 255
        return (v - cv_min) / (cv_max - cv_min) * (cm_max - cm_min) + cm_min

    def cm_to_cv(self, v):
        """color map value to color value"""
        cv_min = self.genren.color_tf[0][0]
        cv_max = self.genren.color_tf[-1][0]
        cm_min = 0
        cm_max = 255
        return (v - cm_min) / (cm_max - cm_min) * (cv_max - cv_min) + cv_min

    def update_gmms(self):
        self.cm_gmm_pts = np.array([v for v in self.genren.color_gmm[:, :]])
        for vs in self.cm_gmm_pts:
            vs[0] = self.cv_to_cm(vs[0])

    def update_plot(self):
        self.update_gmms()
        self.cm_plot.set_data(np.tile(self.genren.color_tf[:, 1:], (6, 1, 1)))
        self.cm_params_plot.set_offsets([(x, 0.5) for x in self.genren.color_gmm[:, 0]])
        self.draw()
        self.main_interface.activateWindow()
        self.main_interface.setFocus()

    def onpress(self, event):
        min_idx, _ = min(enumerate([(x[0] - event.xdata)**2 for x in self.genren.color_gmm]), key=lambda x: x[1])
        self.pt_selected = min_idx
        self.pressed_plot_pt = np.array([event.xdata, event.ydata])
        print('pressed pt:',self.pressed_plot_pt)

    def onmotion(self, event):
        plot_pt = np.array([event.xdata, event.ydata])
        if self.pt_selected is None:
            return
        if plot_pt[0] is None or plot_pt[1] is None:
            return
        if event.button == 1:
            diff_update = np.array([event.xdata, event.ydata]) - self.pressed_plot_pt
            self.genren.color_gmm[self.pt_selected, 0] += diff_update[0]
            self.genren.update_gmm_transfer_function()
            self.update_plot()

        # self.main_interface.do_update()
        self.pressed_plot_pt = plot_pt

    def onrelease(self, event):
        modifiers = QApplication.keyboardModifiers()
        self.modifying_modes = False
        if modifiers == QtCore.Qt.ShiftModifier:
            self.modifying_modes = True

        print('button:',event.button,'old color gmm:',self.genren.color_gmm)
        if event.button == 3:
            color = QColorDialog().getColor()
            self.genren.color_gmm[self.pt_selected, 1:] = np.array([color.redF(), color.greenF(), color.blueF()])
            print(self.genren.color_gmm.shape)
        if self.modifying_modes and event.button == 1:
            color = QColorDialog().getColor()
            new_color = [self.cm_to_cv(event.xdata), color.redF(), color.greenF(), color.blueF()]
            cl = self.genren.color_gmm.tolist()
            cl.append(new_color)
            cl.sort(key=lambda x: x[0])
            self.genren.color_gmm = np.array(cl)
        print('new color gmm:',self.genren.color_gmm)

        self.genren.update_gmm_transfer_function()
        self.main_interface.do_update()
        self.update_plot()
        self.pt_selected = None
        self.pressed_plot_pt = None

class ImageViewer(QWidget):
    def __init__(self, parent=None, genren=None, canvas_size=256, img_res=256):
        super().__init__(parent)
        self.cur_feat = None
        self.main_interface = parent
        self.img_res = img_res
        self.genren = genren
        self.azimuth_delta = 1.0
        self.elevation_delta = 1.0
        self.roll_delta = 1.0
        self.zoom_delta = (self.genren.max_zoom - self.genren.min_zoom) / 100.0
        self.border_color = [64,64,127]
        self.prior_mouse_position = None
        self.pressed_button = None
        self.dragging_region = False
        self.pixmap = QPixmap(self.img_res, self.img_res)
        self.image_label = QLabel()
        self.image_label.setParent(self)
        self.image_label.setGeometry((canvas_size - self.img_res) / 2,
                                     (canvas_size - self.img_res) / 2,
                                     self.img_res,
                                     self.img_res)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.show()

    def post_process_image(self,img,left_pad=4, right_pad=4, bottom_pad=4, top_pad=4):
        for idx in range(3):
            remapped_color = self.border_color[idx]/255.0
            img[idx,:,:left_pad] = remapped_color
            img[idx,:,-right_pad:] = remapped_color
            img[idx,:top_pad,:] = remapped_color
            img[idx,-bottom_pad:,:] = remapped_color

    def save_image(self, filename):
        if self.cur_feat is not None:
            self.genren.opnet_tf_encoding = self.cur_feat
            self.genren.colornet_op_encoding = self.genren.translatenet.opacity_latent_decoder(self.cur_feat)
            predicted_img = self.genren.predict_img(encode_inputs=False)
            self.post_process_image(predicted_img)
            predicted_img *= 255
            predicted_img = predicted_img.astype(np.uint8)
            predicted_rgb_img = np.zeros((predicted_img.shape[1],predicted_img.shape[2],3),dtype=np.uint8)
            for i in range(3):
                predicted_rgb_img[:,:,i] = predicted_img[i,:,:]
            predicted_pil_img = Image.fromarray(predicted_rgb_img)
            predicted_pil_img.save(filename)

    def do_update(self):
        if self.cur_feat is not None:
            self.genren.opnet_tf_encoding = self.cur_feat
            self.genren.colornet_op_encoding = self.genren.translatenet.opacity_latent_decoder(self.cur_feat)
            predicted_img = self.genren.predict_img(encode_inputs=False)
            self.post_process_image(predicted_img)
            bit_img = self.genren.convert_to_bitmap(predicted_img, self.img_res)
            # set the image in top right corner
            q_img = QImage(bit_img.flatten(), self.img_res, self.img_res, QImage.Format_RGB32)
            self.pixmap.convertFromImage(q_img)
            self.image_label.setPixmap(self.pixmap)
            self.image_label.repaint()
            self.repaint()

    def mousePressEvent(self, point):
        self.prior_mouse_position = np.array([point.x(), point.y()])
        modifiers = QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            image_x_pos = self.image_label.pos().x() + self.pos().x()
            image_y_pos = self.image_label.pos().y() + self.pos().y()
            image_pos = self.prior_mouse_position - np.array([image_x_pos, image_y_pos])
            if image_pos[0] >= 0 and image_pos[0] < self.img_res and image_pos[1] >= 0 and image_pos[1] < self.img_res:
                self.dragging_region = True
                self.start_drag_pos = np.array(image_pos)
        self.pressed_button = point.button()

    def mouseReleaseEvent(self, point):
        if self.prior_mouse_position is not None:
            self.main_interface.do_update()
        self.prior_mouse_position = None
        self.pressed_button = None
        self.dragging_region = False

    def mouseMoveEvent(self, point):
        if self.prior_mouse_position is None:
            return

        if self.dragging_region:
            new_mouse_pos = np.array([point.x(), point.y()])
            image_x_pos = self.image_label.pos().x() + self.pos().x()
            image_y_pos = self.image_label.pos().y() + self.pos().y()
            image_pos = new_mouse_pos - np.array([image_x_pos, image_y_pos])
            image_pos[0] = max(0, image_pos[0])
            image_pos[0] = min(self.img_res - 1, image_pos[0])
            image_pos[1] = max(0, image_pos[1])
            image_pos[1] = min(self.img_res - 1, image_pos[1])
            self.do_region_update(self.start_drag_pos, image_pos)
            return

        delta_x = point.x() - self.prior_mouse_position[0]
        delta_y = point.y() - self.prior_mouse_position[1]
        if self.pressed_button == QtCore.Qt.LeftButton:
            self.genren.azimuth -= delta_x * self.azimuth_delta
            if self.genren.azimuth > 360.0:
                self.genren.azimuth -= 360.0
            if self.genren.azimuth < 0:
                self.genren.azimuth += 360.0

            self.genren.elevation += delta_y * self.elevation_delta
            if self.genren.elevation > self.genren.max_elevation:
                self.genren.elevation = self.genren.max_elevation
            if self.genren.elevation < self.genren.min_elevation:
                self.genren.elevation = self.genren.min_elevation

        elif self.pressed_button == QtCore.Qt.RightButton:
            self.genren.zoom -= delta_y * self.zoom_delta
            if self.genren.zoom > self.genren.max_zoom:
                self.genren.zoom = self.genren.max_zoom
            if self.genren.zoom < self.genren.min_zoom:
                self.genren.zoom = self.genren.min_zoom

        self.prior_mouse_position = np.array([point.x(), point.y()])
        self.genren.opnet_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        self.genren.translation_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        self.main_interface.do_update()

def main():
    parser = argparse.ArgumentParser("python feature_explorer.py")
    parser.add_argument("trnet", help="translatenet file name")
    parser.add_argument("--range", default=None, help="npy volume's mix/max range")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--fieldname", default='Scalars_', help='vti scalar field name')
    parser.add_argument("--gid", default=0, type=int, help="GPU device id, default 0")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    renderer = TFExplorer(args)
    print('... tf explorer')
    app.setActiveWindow(renderer)
    print('... set active window')
    sys.exit(app.exec_())
    print('... enter loop')


if __name__ == '__main__':
    main()
