import sys
import os
import time

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QColorDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPalette
from PyQt5.QtSvg import QSvgGenerator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from scipy import spatial
from genren import GenerativeVolumeRenderer

import torch
from torch.autograd import Variable

file_dir = os.path.dirname(os.path.realpath(__file__))
data_generator_dir = os.path.abspath(os.path.join(file_dir, os.pardir)) + '/data_generator'
sys.path.insert(0, data_generator_dir)

import tf_generator

import numpy as np
from PIL import Image
import argparse
# Global variables
matplotlib.rcParams.update({'font.size': 12})
#matplotlib.rcParams['font.weight'] = 'heavy'
matplotlib.rcParams['font.family'] = 'Times New Roman'


class SENExplorer(QWidget):
    def __init__(self, opt=None):
        super().__init__()
        self.num_sens_blocks = 16
        # set Geometry
        self.img_size = 256
        self.tf_width = 512
        self.cm_height = 60
        self.width_pad = 50
        self.height_pad = 0
        self.height = self.img_size + self.cm_height + self.height_pad
        self.width = self.img_size + self.tf_width + self.height_pad + self.width_pad
        self.sens = np.zeros((self.num_sens_blocks, self.num_sens_blocks, 256)).astype(np.uint32)

        # setup genren
        if opt.cuda:
            tr = torch.load(opt.trnet, map_location={'cuda:0': 'cuda:%d' % opt.gid, 'cuda:1': 'cuda:%d' % opt.gid})
        else:
            tr = torch.load(opt.trnet, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu'})
        op = tr.opNet[0]

        if opt.range is None:
            scalar_range = None
        else:
            scalar_range = np.load(opt.range)

        self.genren = GenerativeVolumeRenderer(op, tr, scalar_range=scalar_range, use_cuda=opt.cuda, gid=opt.gid)

        self.cm = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap(name='plasma'))
        # initialize
        self.setWindowTitle('TF Sensitivity')
        self.initUI()
        self.layout_window()

    def update_sensitivities(self):
        sens = self.genren.predict_all_sensitivities(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom, self.num_sens_blocks)
        self.sens = np.abs(sens)
        self.color_map.update_plot(np.min(self.sens), np.max(self.sens))

    def layout_window(self):
        self.image_viewer = ImageViewer(self, self.genren, self.cm, num_blocks=self.num_sens_blocks, op_range=(10, 255))
        self.image_viewer.setGeometry(0, 0, self.img_size, self.img_size)
        self.image_viewer.show()

        self.tf_editor = OTFEditor(self.genren, self, self.tf_width)
        self.tf_editor.setGeometry(self.img_size, 0, self.tf_width, self.img_size)
        self.tf_editor.show()

        self.color_map = MyColorMap(self.cm, parent=self, height=self.cm_height)
        self.color_map.setGeometry(0, self.img_size, self.img_size, self.cm_height)
        self.color_map.show()

        self.ctf_editor = ColorTFEditor(self.genren, self, self.tf_width, self.cm_height)
        self.ctf_editor.setGeometry(self.img_size, self.img_size, self.tf_width, self.cm_height)
        self.ctf_editor.show()

        self.update_sensitivities()
        self.image_viewer.compute_sens_opacity()

    def initUI(self):
        d_geo = QApplication.desktop().screenGeometry()
        d_width = d_geo.width()
        d_height = d_geo.height()
        self.setGeometry(d_width / 2 - self.width / 2, d_height / 2 - self.height / 2, self.width, self.height)
        self.show()

    def update_tf_editor(self, do_full_update=False):
        self.tf_editor.plot_update(do_full_update)

    def update_image(self, do_full_update, update_sens=False):
        self.image_viewer.update_bit_img()
        if (update_sens):
            self.update_sensitivities()
        self.image_viewer.do_update(do_full_update)

    def set_scalar_value(self, sf_val, do_gaussian):
        self.image_viewer.selected_x = sf_val
        self.image_viewer.do_gaussian = do_gaussian
        self.image_viewer.do_update(True)

    def reset_sens_region(self):
        self.image_viewer.do_update(False)

    def keyPressEvent(self, event):
        if (event.key() == QtCore.Qt.Key_Escape):
            sys.exit(0)
        if event.key() == QtCore.Qt.Key_T:
            self.genren.use_opacity_sensitivity=not self.genren.use_opacity_sensitivity
            #self.update_image(True,True)
        if event.key() == QtCore.Qt.Key_Q:
            _, self.genren.color_gmm = tf_generator.generate_opacity_color_gmm(self.genren.min_scalar_value, self.genren.max_scalar_value, 1)
            self.genren.update_gmm_transfer_function()
            self.update_image(True)
            self.ctf_editor.update_plot()
        if event.key() == QtCore.Qt.Key_R:
            pass
            #self.do_update()
        if event.key() == QtCore.Qt.Key_C:
            np.save('cached_opacity.npy',self.genren.opacity_gmm)
            np.save('cached_color.npy',self.genren.color_gmm)
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
            self.tf_editor.save_fig(generate_unique_filename('tf_editor','svg'))
            self.color_map.save_fig(generate_unique_filename('cm','svg'))
            self.ctf_editor.save_fig(generate_unique_filename('ctf_editor','svg'))
            self.image_viewer.save_image(generate_unique_filename('vimage','png'), generate_unique_filename('simage','png'))

#

class ImageViewer(QWidget):
    def __init__(self, parent, genren, cm, img_res=256, num_blocks=8, op_range=(127, 255)):
        super().__init__(parent)
        self.num_blocks = num_blocks
        self.main_interface = parent
        self.img_res = img_res
        self.genren = genren
        self.prior_mouse_position = None
        self.pressed_button = None
        self.pixmap = QPixmap(self.img_res, self.img_res)
        self.image_label = QLabel()
        self.image_label.setParent(self)
        self.image_label.setGeometry(0, 0, self.img_res, self.img_res)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.show()
        self.sens = np.zeros([self.num_blocks, self.num_blocks])
        self.op_range = op_range
        self.g_sens_cm = cm

        self.np_img = None
        self.selected_x = 0
        self.do_gaussian = False

        # set background color
        pal = QPalette()
        bgColor = QColor()
        # bgColor.setRgbF(0.310999694819562063, 0.3400015259021897, 0.4299992370489052)
        bgColor.setRgbF(0, 0, 0)
        pal.setColor(QPalette.Background, bgColor)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    def compute_sens_opacity(self):
        x_samples = np.linspace(self.genren.min_scalar_value,self.genren.max_scalar_value,num=self.main_interface.sens.shape[1])
        if self.do_gaussian:
            gaussian_weights = np.exp(-(x_samples-self.selected_x)**2/self.genren.min_bandwidth**2)
        else:
            gaussian_weights = np.exp(-(x_samples-self.selected_x)**2/self.genren.min_bandwidth**2)
            argmax = np.argmax(gaussian_weights)
            gaussian_weights = np.zeros(x_samples.shape[0])
            gaussian_weights[argmax] = 1.0
        gaussian_sum = np.sum(gaussian_weights)
        self.sens = self.main_interface.sens.dot(gaussian_weights) / gaussian_sum

        self.l_max = np.max(self.sens)
        self.l_min = np.min(self.sens)
        l_range = self.l_max-self.l_min
        op_range = self.op_range[1]-self.op_range[0]

        block_size = self.img_res // self.num_blocks
        opacity = np.zeros((self.img_res, self.img_res)).astype(np.float32)
        for r in range(self.num_blocks):
            for c in range(self.num_blocks):
                idx = c * self.num_blocks + r
                opacity[c*block_size:(c+1)*block_size,r*block_size:(r+1)*block_size] = self.op_range[0] + (1 - (self.sens[idx] - self.l_min) / l_range) * op_range
            #
        #
        return opacity / 255

    def update_bit_img(self):
        self.np_img = self.genren.predict_img()

    def get_g_sens_color(self, v):
        if self.g_max == self.g_min:
            return np.array([0, 0, 0])
        r = (v - self.g_min) / (self.g_max - self.g_min)
        c = self.g_sens_cm.to_rgba(r, norm=False)
        return np.array([c[0], c[1], c[2]])

    def blend_color(self, rgb_img, op_img):
        start = time.time()
        all_vals = self.main_interface.sens.flatten()
        sorted_vals = np.sort(all_vals)
        print('top 20 vals:',sorted_vals[-20:])
        self.g_max = np.max(self.main_interface.sens)
        self.g_min = np.min(self.main_interface.sens)

        # compute global sensitivity
        block_size = self.img_res // self.num_blocks
        img = np.zeros((3,self.img_res,self.img_res)).astype(np.float)
        print('rgb image size:',rgb_img.shape,'op image size:',op_img.shape)

        for r in range(self.num_blocks):
            for c in range(self.num_blocks):
                idx = c * self.num_blocks + r
                sensitivity_color = self.get_g_sens_color(self.sens[idx]).reshape((3,1,1))
                rgb_block = rgb_img[:,c*block_size:(c+1)*block_size,r*block_size:(r+1)*block_size]
                op_block = np.tile(op_img[c*block_size:(c+1)*block_size,r*block_size:(r+1)*block_size],(3,1,1))
                sensitivity_block = np.ones(rgb_block.shape)*sensitivity_color
                img[:,c*block_size:(c+1)*block_size,r*block_size:(r+1)*block_size] = rgb_block*op_block + sensitivity_block*(1.0-op_block)
            #
        #
        return img

    def do_update(self, update_sens):
        if update_sens:
            np_opacity = self.compute_sens_opacity()
        else:
            np_opacity = np.ones((self.img_res, self.img_res)).astype(np.float32)

        if self.np_img is None:
            self.update_bit_img()
        img = self.blend_color(self.np_img, np_opacity)
        bit_img = self.genren.convert_to_bitmap(img, self.img_res)

        self.q_img = QImage(bit_img.flatten(), self.img_res, self.img_res, QImage.Format_RGB32)
        self.pixmap.convertFromImage(self.q_img)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.repaint()
        self.repaint()

    def mousePressEvent(self, point):
        self.prior_mouse_position = np.array([point.x(), point.y()])
        self.pressed_button = point.button()

    def mouseReleaseEvent(self, point):
        if self.prior_mouse_position is not None:
            self.update_bit_img()
            self.main_interface.update_sensitivities()
            self.do_update(False)
        self.prior_mouse_position = None
        self.pressed_button = None
        self.dragging_region = False

    def mouseMoveEvent(self, point):
        if self.prior_mouse_position is None:
            return

        delta_x = point.x() - self.prior_mouse_position[0]
        delta_y = point.y() - self.prior_mouse_position[1]
        if self.pressed_button == QtCore.Qt.LeftButton:
            self.genren.azimuth -= delta_x * self.genren.azimuth_delta
            if self.genren.azimuth > 360.0:
                self.genren.azimuth -= 360.0
            if self.genren.azimuth < 0:
                self.genren.azimuth += 360.0

            self.genren.elevation += delta_y * self.genren.elevation_delta
            if self.genren.elevation > self.genren.max_elevation:
                self.genren.elevation = self.genren.max_elevation
            if self.genren.elevation < self.genren.min_elevation:
                self.genren.elevation = self.genren.min_elevation

        elif self.pressed_button == QtCore.Qt.RightButton:
            self.genren.zoom -= delta_y * self.genren.zoom_delta
            if self.genren.zoom > self.genren.max_zoom:
                self.genren.zoom = self.genren.max_zoom
            if self.genren.zoom < self.genren.min_zoom:
                self.genren.zoom = self.genren.min_zoom

        print('elevation:',self.genren.elevation,'azimuth:',self.genren.azimuth,'zoom:',self.genren.zoom)
        self.prior_mouse_position = np.array([point.x(), point.y()])
        #self.genren.opnet_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        #self.genren.translation_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        self.np_img = self.genren.predict_sensitivity(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom, return_img=True)
        self.main_interface.update_tf_editor(True)
        self.do_update(False)

    def save_image(self, original_filename, sensitivity_filename):
        def do_save_img(img, filename):
            out_img = 255*img
            out_img = out_img.astype(np.uint8)
            out_rgb_img = np.zeros((out_img.shape[1],out_img.shape[2],3),dtype=np.uint8)
            for i in range(3):
                out_rgb_img[:,:,i] = out_img[i,:,:]
            predicted_pil_img = Image.fromarray(out_rgb_img)
            predicted_pil_img.save(filename)
        #
        do_save_img(self.np_img, original_filename)
        do_save_img(self.blend_color(self.np_img, self.compute_sens_opacity()), sensitivity_filename)
    #
#

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=256, height=256, dpi=100):
        # self.fig = plt.figure(frameon=False)
        self.fig, self.axis = plt.subplots(figsize=(width, height))
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)
        self.compute_initial_figure()

    def compute_initial_figure(self):
        pass

    def save_fig(self, fig_filename):
        self.fig.savefig(fig_filename)


class MyColorMap(MyMplCanvas):
    def __init__(self, cm, *args, **kwargs):
        self.cm = cm
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.min = 0
        self.max = 1

    def compute_initial_figure(self):
        self.axis.set_title('Global Sensitivity CM')
        x = np.linspace(0, 1, 256)
        colors = self.cm.to_rgba(x)
        colors = colors[:, :3]
        colors = np.tile(colors, (16, 1, 1))
        self.axis.imshow(colors)
        self.axis.get_yaxis().set_ticks([])
        self.axis.get_xaxis().set_ticks([0, 127, 255])
        self.axis.get_xaxis().set_ticklabels([0, 0.5, 1.0])

    def update_plot(self, min_v, max_v):
        min_l = "%.2f" % min_v
        max_l = "%.2f" % max_v
        med_l = "%.2f" % ((max_v + min_v) / 2)
        self.axis.get_xaxis().set_ticklabels([min_l, med_l, max_l])
        self.draw()


class OTFEditor(MyMplCanvas):
    def __init__(self, genren, *args, **kwargs):
        self.tf_line_width = 1
        self.tf_dot_size = 12
        self.selection_thresh = 0.05

        self.genren = genren
        self.main_interface = args[0]

        self.recent_mean = np.array([0.5, 0.5])

        MyMplCanvas.__init__(self, *args, **kwargs)
        self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('key_press_event', self.keypressed)
        self.fig.canvas.mpl_connect('key_release_event', self.keyreleased)
        self.pt_selected = None
        self.pressed_plot_pt = None
        self.modifying_modes = False
        self.adjusting_bandwidth = False
        self.selected_x = None
        self.selected_range = False

    def keypressed(self):
        pass

    def keyreleased(self):
        pass

    def update_gmms(self):
        self.fig_x_diam = self.genren.opacity_tf[-1, 0] - self.genren.opacity_tf[0, 0]
        self.gmm_pts = []
        for mean in self.genren.opacity_gmm[:, 0]:
            gmm_val = np.sum(self.genren.opacity_gmm[:, 2] * np.exp(-np.power(-(mean - self.genren.opacity_gmm[:, 0]), 2) / np.power(self.genren.opacity_gmm[:, 1], 2)))
            self.gmm_pts.append([mean, gmm_val])
        self.gmm_pts = np.array(self.gmm_pts)

    def compute_otf_axis(self):
        self.genren.predict_sensitivity(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        tf_sensitivity = np.abs(self.genren.predicted_sensitivity)
        self.axis.set_ylim([0.001, np.max(tf_sensitivity)])
        self.axis.plot(self.genren.opacity_tf[:, 0], tf_sensitivity, linewidth=1, alpha=0.6, color='crimson')
        self.axis.fill_between(self.genren.opacity_tf[:, 0], 0, tf_sensitivity, facecolor='pink', alpha=0.5)
        self.axis.set_ylabel('TF Sensitivity', color='crimson')
        self.tf_axes = self.axis.twinx()

        self.tf_plot, = self.tf_axes.plot(self.genren.opacity_tf[:, 0], self.genren.opacity_tf[:, 1], linewidth=self.tf_line_width, color='darkblue')
        self.tf_axes.set_ylim([0, 1])
        self.tf_axes.set_ylabel('Opacity TF', color='darkblue')
        self.tf_axes.yaxis.tick_left()
        self.tf_axes.yaxis.set_label_position('left')
        self.axis.yaxis.tick_right()
        self.axis.yaxis.set_label_position('right')
        self.tf_params_plot = self.tf_axes.scatter(self.gmm_pts[:, 0], self.gmm_pts[:, 1], s=self.tf_dot_size)
        self.axis.set_xlim([self.min_data, self.max_data])

    def compute_initial_figure(self):
        self.min_data = np.min(self.genren.opacity_tf[:, 0])
        self.max_data = np.max(self.genren.opacity_tf[:, 0])
        self.select_radius = 0.5*self.genren.gaussian_range
        self.update_gmms()
        self.compute_otf_axis()

    def form_gaussian_cm(self):
        x_min = np.max([self.selected_x-0.5*self.genren.gaussian_range, self.min_data])
        x_max = np.min([self.selected_x+0.5*self.genren.gaussian_range, self.max_data])
        n_steps = 50
        x_eps = (x_max-x_min)/(100.0*n_steps)
        x_vals = np.arange(x_min,x_max+x_eps,(x_max-x_min)/n_steps)
        alphas = np.exp(-(x_vals-self.selected_x)**2/self.genren.min_bandwidth**2)
        normalized_vals = (x_vals - x_min) / (x_max-x_min)
        normalized_vals[0] = 0
        normalized_vals[-1] = 1
        cdict = dict()
        cdict['red'] = [(x, 204/255., 204/255.) for x in normalized_vals]
        cdict['green'] = [(x, 102/255., 102/255.) for x in normalized_vals]
        cdict['blue'] = [(x, 102/255., 102/255.) for x in normalized_vals]
        cdict['alpha'] = [(x, a, a) for x,a in zip(normalized_vals,alphas)]
        return LinearSegmentedColormap('gaussian_cm', cdict)

    def plot_update(self, do_full_update=False):
        self.update_gmms()
        self.tf_plot.set_ydata(self.genren.opacity_tf[:, 1])
        self.tf_params_plot.set_offsets(self.gmm_pts)
        # self.cm_plot.set_data(np.tile(self.genren.color_tf[:, 1:], (16, 1, 1)))

        if do_full_update:
            self.axis.cla()
            tf_sensitivity = np.abs(self.genren.predicted_sensitivity)
            # self.axis.set_ylim([0,np.max(tf_sensitivity)])
            self.axis.set_ylim([0, np.max(tf_sensitivity)])
            self.axis.plot(self.genren.opacity_tf[:, 0], tf_sensitivity, linewidth=1, alpha=0.6, color='crimson')
            self.axis.fill_between(self.genren.opacity_tf[:, 0], 0, tf_sensitivity, facecolor='pink', alpha=0.5)
            self.axis.set_ylabel('Sensitivity', color='crimson')
            self.axis.yaxis.tick_right()
            self.axis.yaxis.set_label_position('right')
            self.axis.set_xlim([self.min_data, self.max_data])

            min_sense,max_sense = np.min(tf_sensitivity),np.max(tf_sensitivity)
            self.tf_plot.set_ydata(self.genren.opacity_tf[:, 1])
            self.tf_params_plot.set_offsets(self.gmm_pts)
            if self.selected_x is not None:
                if self.selected_range:
                    x1 = min([self.selected_x - self.select_radius, self.max_data])
                    x1 = max([x1, self.min_data])
                    x2 = min([self.selected_x + self.select_radius, self.max_data])
                    x2 = max([x2, self.min_data])

                    im_res = tf_sensitivity.shape[0]
                    sense_field = np.ones((im_res,im_res))*np.linspace(min_sense,max_sense,num=im_res)

                    sampled_tf_sensitivity = np.zeros((im_res,im_res))
                    for rdx in range(im_res):
                        sense_x = x1+(x2-x1)*(rdx/im_res)
                        sense_ind = (im_res-1)*(sense_x-self.min_data)/(self.max_data-self.min_data)
                        if sense_ind >= (im_res-1):
                            sampled_tf_sensitivity[rdx] = tf_sensitivity[im_res-1]
                        else:
                            x_alpha = sense_ind - int(sense_ind)
                            sampled_tf_sensitivity[rdx] = (1-x_alpha)*tf_sensitivity[int(sense_ind)] + x_alpha*tf_sensitivity[int(sense_ind+1)]

                    sampled_sense_image = np.ones((im_res,im_res))*sampled_tf_sensitivity
                    thresh_field = np.zeros((im_res,im_res,4))
                    thresh_field[sense_field.T>sampled_sense_image.T] = 1
                    thresh_field = np.flip(thresh_field,0)
                    self.axis.imshow(sense_field, cmap=self.form_gaussian_cm(), interpolation='bicubic', extent=(x1,x2,0,np.max(tf_sensitivity)), aspect='auto')
                    self.axis.imshow(thresh_field, extent=(x1,x2,0,np.max(tf_sensitivity)), aspect='auto')
                else:
                    self.axis.axvline(x=self.selected_x, linewidth=self.tf_line_width, color='k')
        self.draw()
        self.main_interface.activateWindow()
        self.main_interface.setFocus()

    def onpress(self, event):
        # find closest mean - if close enough, then indicate we are updating
        plot_pt = np.array([event.xdata, event.ydata])
        if plot_pt[0] is None:
            return

        closest_mean_ind = np.argmin(np.linalg.norm(plot_pt - self.gmm_pts, axis=1))
        mean_pt = self.gmm_pts[closest_mean_ind, :]
        if np.linalg.norm(mean_pt - plot_pt) / self.fig_x_diam < self.selection_thresh:
            self.pt_selected = closest_mean_ind
            self.pressed_plot_pt = plot_pt

    def onmotion(self, event):
        plot_pt = np.array([event.xdata, event.ydata])
        if plot_pt[0] is None:
            return
        if QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
            self.selected_x = event.xdata
            self.selected_range = False
            self.main_interface.set_scalar_value(self.selected_x,self.selected_range)
        elif QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            self.selected_x = event.xdata
            self.selected_range = True
            self.main_interface.set_scalar_value(self.selected_x,self.selected_range)
        else:
            self.selected_x = None
            self.selected_range = False
            self.main_interface.reset_sens_region()

        closest_ind = np.argmin(np.absolute(plot_pt[0] - self.genren.opacity_tf[:, 0]))
        mean_pt = self.genren.opacity_tf[closest_ind, 0]
        self.recent_mean = [mean_pt, mean_pt]

        if self.pt_selected is None:
            self.plot_update(True)
            self.draw()
            return

        if event.button == 1:
            diff_update = plot_pt - self.pressed_plot_pt
            self.genren.opacity_gmm[self.pt_selected, 0] += diff_update[0]
            self.genren.opacity_gmm[self.pt_selected, 2] += diff_update[1]
        elif event.button == 3:
            diff_bandwidth_update = (plot_pt[0] - self.pressed_plot_pt[0])
            new_bandwidth = self.genren.opacity_gmm[self.pt_selected, 1] + diff_bandwidth_update
            new_bandwidth = min(new_bandwidth, self.genren.max_bandwidth)
            new_bandwidth = max(new_bandwidth, self.genren.min_bandwidth)
            print('old bandwidth:', self.genren.opacity_gmm[self.pt_selected, 1], 'new bandwidth:', new_bandwidth)
            self.genren.opacity_gmm[self.pt_selected, 1] = new_bandwidth
        self.genren.update_gmm_transfer_function()
        self.genren.encode_inputs()
        self.main_interface.update_image(False)
        self.plot_update()
        self.pressed_plot_pt = plot_pt

    def onrelease(self, event):
        modifiers = QApplication.keyboardModifiers()
        self.modifying_modes = False
        if modifiers == QtCore.Qt.ShiftModifier:
            self.modifying_modes = True

        plot_pt = np.array([event.xdata, event.ydata])
        do_main_update = False
        if self.modifying_modes and plot_pt[0] is not None:
            if event.button == 1:
                new_pt = np.array([plot_pt[0], self.genren.min_bandwidth, plot_pt[1]])
                print('new point:', new_pt)
                self.genren.opacity_gmm = np.vstack([self.genren.opacity_gmm, new_pt])
                do_main_update = True
            elif event.button == 3:
                closest_mean_ind = np.argmin(np.linalg.norm(plot_pt - self.gmm_pts, axis=1))
                mean_pt = self.gmm_pts[closest_mean_ind, :]
                if np.linalg.norm(mean_pt - plot_pt) / self.fig_x_diam < self.selection_thresh:
                    subset_inds = np.arange(self.genren.opacity_gmm.shape[0]) != closest_mean_ind
                    self.genren.opacity_gmm = self.genren.opacity_gmm[subset_inds, :]
                    do_main_update = True

        # if do_main_update:
        self.genren.update_gmm_transfer_function()
        self.genren.encode_inputs()
        self.genren.predict_sensitivity(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        self.main_interface.update_image(False, True)
        self.plot_update(True)

        self.pt_selected = None
        self.pressed_plot_pt = None


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
        self.fig.suptitle('Color TF')

    def compute_initial_figure(self):
        self.update_gmms()
        cm_data = np.tile(self.genren.color_tf[:, 1:], (15, 1, 1))
        self.cm_params_plot = self.axis.scatter(self.cm_gmm_pts[:, 0],
                                                [7 for i in range(self.cm_gmm_pts.shape[0])],
                                                s=self.tf_dot_size)
        self.cm_plot = self.axis.imshow(cm_data)

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
        self.cm_plot.set_data(np.tile(self.genren.color_tf[:, 1:], (15, 1, 1)))
        self.cm_params_plot.set_offsets([(x, 7) for x in self.cm_gmm_pts[:, 0]])
        self.draw()
        self.main_interface.activateWindow()
        self.main_interface.setFocus()

    def onpress(self, event):
        min_idx, _ = min(enumerate([(x[0] - event.xdata)**2 for x in self.cm_gmm_pts]), key=lambda x: x[1])
        self.pt_selected = min_idx
        self.pressed_plot_pt = np.array([event.xdata, event.ydata])
        print(self.pressed_plot_pt)

    def onmotion(self, event):
        plot_pt = np.array([event.xdata, event.ydata])
        if self.pt_selected is None:
            return
        if plot_pt[0] is None or plot_pt[1] is None:
            return
        if event.button == 1:
            diff_update = np.array([event.xdata, event.ydata]) - self.pressed_plot_pt
            diff = self.cm_to_cv(diff_update[0])
            self.genren.color_gmm[self.pt_selected, 0] += diff
            self.genren.update_gmm_transfer_function()
            self.update_plot()

        # self.main_interface.do_update()
        self.pressed_plot_pt = plot_pt

    def onrelease(self, event):
        modifiers = QApplication.keyboardModifiers()
        self.modifying_modes = False
        if modifiers == QtCore.Qt.ShiftModifier:
            self.modifying_modes = True

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

        self.genren.update_gmm_transfer_function()
        self.genren.encode_inputs()
        self.main_interface.update_image(False)
        self.update_plot()
        self.pt_selected = None
        self.pressed_plot_pt = None


def main():
    parser = argparse.ArgumentParser("python sensitivity.py")
    parser.add_argument("trnet", help="translatenet file name")
    parser.add_argument("--range", default=None, help="npy volume's mix/max range")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gid", default=0, type=int, help="GPU device id, default 0")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    renderer = SENExplorer(args)
    app.setActiveWindow(renderer)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
