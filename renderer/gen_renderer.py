import sys
import os

import torch
from genren import GenerativeVolumeRenderer
from tfgui import OpacityTFWidget
import numpy as np
import time
import vtk
from vtk_renderer import VTKRenderer

from PIL import Image
from tqdm import tqdm

file_dir = os.path.dirname(os.path.realpath(__file__))
data_generator_dir = os.path.abspath(os.path.join(file_dir, os.pardir)) + '/data_generator'
sys.path.insert(0, data_generator_dir)

import tf_generator

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QObject, QPoint, QPointF, QTimer
from PyQt5.QtGui import QImage, QGuiApplication, QScreen, QPixmap, QPainter, QPen, QLinearGradient, QGradient, QColor
from PyQt5.QtWidgets import QFrame, QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import argparse

parser = argparse.ArgumentParser("python gen_renderer.py")
parser.add_argument("vol", help="volume (vti) file name")
parser.add_argument("trnet", help="translatenet file name")
parser.add_argument("--cuda", action="store_true", help="use cuda")
parser.add_argument("--gid", default=0, type=int, help="gpu device id default 0")

args = parser.parse_args()
volume_filename = args.vol
translatenet_filename = args.trnet

sf_name = 'ImageFile'
sf_name = 'Scalars_'
volume_dataset_reader = vtk.vtkXMLImageDataReader()
volume_dataset_reader.SetFileName(volume_filename)
volume_dataset_reader.Update()
volume_data = volume_dataset_reader.GetOutput()
volume_data.GetPointData().SetActiveAttribute(sf_name, 0)
data_range = volume_data.GetPointData().GetScalars().GetRange()

matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['font.family'] = 'Times New Roman'

use_vtk = True

class NeuralVolumeRenderer(QWidget):
    def __init__(self):
        super().__init__()

        if args.cuda:
            self.translatenet = torch.load(args.trnet, map_location={'cuda:0': 'cuda:%d' % args.gid, 'cuda:1': 'cuda:%d' % args.gid})
        else:
            self.translatenet = torch.load(args.trnet, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu'})
        self.opnet = self.translatenet.opNet[0]
        self.genren = GenerativeVolumeRenderer(self.opnet, self.translatenet, scalar_range=np.array([data_range[0],data_range[1]]), use_cuda=args.cuda)

        self.saving_imgs = False
        self.prior_mouse_position = None
        self.pressed_button = None
        self.use_placeholder = False

        self.dragging_region = False

        self.target_res = 256

        self.opacity_tf_width = 1.75 * self.target_res
        #self.opacity_tf_height = 350
        self.opacity_tf_height = self.target_res
        self.widget_width = self.opacity_tf_width
        #self.widget_height = self.target_res+self.opacity_tf_height
        self.widget_height = self.target_res

        self.main_layout = QHBoxLayout()
        self.main_layout.setSpacing(5.0)
        self.setLayout(self.main_layout)

        self.view_widget = QWidget()
        self.view_layout = QHBoxLayout()
        # self.view_layout.setSpacing(0.0)
        self.view_widget.setLayout(self.view_layout)
        self.view_widget.resize(self.target_res, self.target_res)

        # volume rendering view
        self.q_pixmap = QPixmap(self.target_res, self.target_res)
        self.image_label = QLabel()
        self.image_label.setPixmap(self.q_pixmap)

        # placeholder view (optional)
        self.placeholder_pixmap = QPixmap(self.target_res, self.target_res)
        self.placeholder_label = QLabel()
        self.placeholder_label.setPixmap(self.placeholder_pixmap)

        # organize them
        self.view_layout.addWidget(self.image_label)
        if self.use_placeholder:
            self.view_layout.addWidget(self.placeholder_label)
        self.view_widget.updateGeometry()

        self.main_layout.addWidget(self.view_widget)
        self.main_layout.setAlignment(self.view_widget, QtCore.Qt.AlignVCenter)

        # opacity function view
        self.opacity_tf_view = OpacityTFWidget(self.view_widget)
        self.opacity_tf_view.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.opacity_tf_view.setFocus()
        self.opacity_tf_view.setup_widget(self.genren, self)
        self.opacity_tf_view.setFixedSize(self.opacity_tf_width, self.opacity_tf_height)
        self.main_layout.addWidget(self.opacity_tf_view)
        self.main_layout.setAlignment(self.opacity_tf_view, QtCore.Qt.AlignVCenter)

        self.initUI()
        self.activateWindow()
        self.setFocus()

        # setup vtk if we are using it
        if use_vtk:
            self.vtk_renderer = VTKRenderer(volume_data, self.genren)

        self.setWindowTitle('Generative Model')
        self.genren.opnet_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        self.genren.translation_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        self.do_update()

    def initUI(self):
        self.setGeometry(500, 500, self.widget_width, self.widget_height)
        self.show()

    def do_region_update(self, start_region, end_region):
        min_x = min(start_region[0], end_region[0])
        min_y = min(start_region[1], end_region[1])
        max_x = max(start_region[0], end_region[0])
        max_y = max(start_region[1], end_region[1])
        min_bb = np.array([min_x, min_y])
        max_bb = np.array([max_x, max_y])

        bit_img = self.genren.convert_to_bitmap(self.predicted_img, self.target_res, min_bb=min_bb, max_bb=max_bb)
        self.q_img = QImage(bit_img.flatten(), self.target_res, self.target_res, QImage.Format_RGB32)
        self.q_pixmap.convertFromImage(self.q_img)
        self.image_label.setPixmap(self.q_pixmap)
        self.repaint()

    def do_update(self):
        self.predicted_img = self.genren.predict_img()

        bit_img = self.genren.convert_to_bitmap(self.predicted_img, self.target_res)
        self.q_img = QImage(bit_img.flatten(), self.target_res, self.target_res, QImage.Format_RGB32)
        self.q_pixmap.convertFromImage(self.q_img)
        self.image_label.setPixmap(self.q_pixmap)
        self.repaint()

        if use_vtk:
            self.vtk_renderer.do_render(self.genren.elevation, self.genren.azimuth, self.genren.zoom)

        '''
        self.activateWindow()
        self.setFocus()
        self.opacity_tf_view.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.opacity_tf_view.setFocus()
        '''

        if self.saving_imgs:
            self.snap_ind = 0
            while True:
                snap_name = 'snap' + str(self.snap_ind) + '.png'
                if os.path.exists(snap_name):
                    self.snap_ind += 1
                else:
                    break

            if self.snap_ind == 0:
                placeholder_x_pos = self.placeholder_label.pos().x() + self.view_widget.pos().x()
                placeholder_y_pos = self.placeholder_label.pos().y() + self.view_widget.pos().y()
                print('this is where you can stick it:', placeholder_x_pos, placeholder_y_pos)

            # image
            cur_pixmap = QPixmap(self.size())
            self.render(cur_pixmap)
            cur_pixmap.save(snap_name)

            # params
            params_name = 'params' + str(self.snap_ind) + '.npy'
            view_params = np.array([self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom])
            vis_params = np.concatenate((view_params, self.genren.opacity_tf.ravel(), self.genren.color_tf.ravel()))
            np.save(params_name, vis_params)

    def keyPressEvent(self, event):
        print('key press:', event)
        if event.key() == QtCore.Qt.Key_R:
            self.saving_imgs = not self.saving_imgs
        elif event.key() == QtCore.Qt.Key_Q:
            _, self.genren.color_gmm = tf_generator.generate_opacity_color_gmm(self.genren.min_scalar_value, self.genren.max_scalar_value, 1)
            self.genren.update_gmm_transfer_function()
            self.opacity_tf_view.plot_update(True)
            self.do_update()
        elif event.key() == QtCore.Qt.Key_S:
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

            # plot
            self.opacity_tf_view.save_fig(generate_unique_filename('sensitivity','svg'))
            # synthesized image
            self.genren.save_image(generate_unique_filename('vimage','png'))

        elif event.key() == QtCore.Qt.Key_Z:
            self.genren.toggle_zero_view()
            self.genren.translation_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)

        self.do_update()

    def mousePressEvent(self, point):
        self.prior_mouse_position = np.array([point.x(), point.y()])
        modifiers = QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            image_x_pos = self.image_label.pos().x() + self.view_widget.pos().x()
            image_y_pos = self.image_label.pos().y() + self.view_widget.pos().y()
            image_pos = self.prior_mouse_position - np.array([image_x_pos, image_y_pos])
            if image_pos[0] >= 0 and image_pos[0] < self.target_res and image_pos[1] >= 0 and image_pos[1] < self.target_res:
                self.dragging_region = True
                self.start_drag_pos = np.array(image_pos)
        self.pressed_button = point.button()

    def mouseReleaseEvent(self, point):
        if self.prior_mouse_position is not None:
            if self.dragging_region:
                new_mouse_pos = np.array([point.x(), point.y()])
                image_x_pos = self.image_label.pos().x() + self.view_widget.pos().x()
                image_y_pos = self.image_label.pos().y() + self.view_widget.pos().y()
                image_pos = new_mouse_pos - np.array([image_x_pos, image_y_pos])
                image_pos[0] = max(0, image_pos[0])
                image_pos[0] = min(self.target_res - 1, image_pos[0])
                image_pos[1] = max(0, image_pos[1])
                image_pos[1] = min(self.target_res - 1, image_pos[1])

                min_x = min(self.start_drag_pos[0], image_pos[0])
                min_y = min(self.start_drag_pos[1], image_pos[1])
                max_x = max(self.start_drag_pos[0], image_pos[0])
                max_y = max(self.start_drag_pos[1], image_pos[1])
                min_bb = np.array([min_x, min_y])
                max_bb = np.array([max_x, max_y])

                print('min bb:',min_bb,'max bb:',max_bb)
                self.genren.predict_sensitivity(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom, min_bb, max_bb)
            else:
                self.genren.predict_sensitivity(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
            self.opacity_tf_view.plot_update(True)
        self.prior_mouse_position = None
        self.pressed_button = None
        self.dragging_region = False

    def mouseMoveEvent(self, point):
        if self.prior_mouse_position is None:
            return

        if self.dragging_region:
            new_mouse_pos = np.array([point.x(), point.y()])
            image_x_pos = self.image_label.pos().x() + self.view_widget.pos().x()
            image_y_pos = self.image_label.pos().y() + self.view_widget.pos().y()
            image_pos = new_mouse_pos - np.array([image_x_pos, image_y_pos])
            image_pos[0] = max(0, image_pos[0])
            image_pos[0] = min(self.target_res - 1, image_pos[0])
            image_pos[1] = max(0, image_pos[1])
            image_pos[1] = min(self.target_res - 1, image_pos[1])
            self.do_region_update(self.start_drag_pos, image_pos)
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
            '''
            self.roll -= delta_x*self.roll_delta
            if self.roll > max_roll:
                self.roll = max_roll
            if self.roll < min_roll:
                self.roll = min_roll
            '''

        self.prior_mouse_position = np.array([point.x(), point.y()])
        start = time.time()
        self.genren.opnet_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        self.genren.translation_encode_view(self.genren.elevation, self.genren.azimuth, self.genren.roll, self.genren.zoom)
        end = time.time()
        self.do_update()
        # self.opacity_tf_view.plot_update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    renderer = NeuralVolumeRenderer()
    app.setActiveWindow(renderer)
    sys.exit(app.exec_())
