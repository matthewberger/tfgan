from genren import GenerativeVolumeRenderer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QColorDialog


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=200, height=5, dpi=100):
        self.fig = plt.figure()
        self.axes = []
        self.axes.append(self.fig.add_subplot(2, 1, 1))
        self.axes.append(self.fig.add_subplot(2, 1, 2))

        self.fig.set_figheight(height)
        self.compute_initial_figure()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.updateGeometry(self)
    #

    def compute_initial_figure(self):
        pass
    #
#


class OpacityTFWidget(MyMplCanvas):
    def __init__(self, *args, **kwargs):
        self.has_renderer = False
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)

        self.pt_selected = None
        self.pressed_plot_pt = None
        self.modifying_modes = False
        self.adjusting_bandwidth = False

        self.tf_line_width = 1
        self.tf_dot_size = 12
        self.cm_dot_size = 16

    def setup_widget(self, genren, main_interface):
        self.genren = genren
        self.main_interface = main_interface
        self.has_renderer = True
        self.compute_initial_figure()
        self.min_bandwidth = self.genren.scalar_range * 0.05
        self.max_bandwidth = self.genren.scalar_range * 0.4
        self.bandwidth_delta = self.genren.scalar_range * 0.01

    def compute_initial_figure(self):
        self.selection_thresh = 0.05
        if self.has_renderer:
            self.update_gmms()
            self.op_plot, = self.axes[0].plot(self.genren.opacity_tf[:, 0],
                                              self.genren.opacity_tf[:, 1], linewidth=self.tf_line_width)
            # self.tf_cursor, = self.axes[0].plot([],[], linewidth=self.tf_line_width/2, linestyle='dashed')
            self.op_params_plot = self.axes[0].scatter(self.op_gmm_pts[:, 0], self.op_gmm_pts[:, 1], s=self.tf_dot_size)
            # print('gmm x values:',self.op_gmm_pts[:,0])
            self.axes[0].set_ylim([0, 1])

            cm_data = np.tile(self.genren.color_tf[:, 1:], (16, 1, 1))
            self.cm_params_plot = self.axes[1].scatter(
                self.cm_gmm_pts[:, 0], [8 for i in range(self.cm_gmm_pts.shape[0])],
                self.cm_dot_size)
            self.cm_plot = self.axes[1].imshow(cm_data)
            self.axes[1].axis('off')
            # plt.tight_layout()

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
        self.fig_x_diam = self.genren.opacity_tf[-1, 0] - self.genren.opacity_tf[0, 0]
        self.op_gmm_pts = []
        for mean in self.genren.opacity_gmm[:, 0]:
            op_gmm_vals = np.sum(self.genren.opacity_gmm[:, 2] * np.exp(-np.power(-(mean - self.genren.opacity_gmm[:, 0]), 2) / np.power(self.genren.opacity_gmm[:, 1], 2)))
            self.op_gmm_pts.append([mean, op_gmm_vals])
        self.op_gmm_pts = np.array(self.op_gmm_pts)
        self.cm_gmm_pts = np.array([v for v in self.genren.color_gmm[:, :]])
        for vs in self.cm_gmm_pts:
            vs[0] = self.cv_to_cm(vs[0])

    def plot_update(self, do_full_update=False):
        self.update_gmms()
        self.op_plot.set_ydata(self.genren.opacity_tf[:, 1])
        self.op_params_plot.set_offsets(self.op_gmm_pts)
        self.cm_plot.set_data(np.tile(self.genren.color_tf[:, 1:], (16, 1, 1)))
        self.cm_params_plot.set_offsets([(x, 8) for x in self.cm_gmm_pts[:, 0]])

        # start = time.time()
        self.draw()
        # end = time.time()
        # print('draw time:', (end - start))

        self.main_interface.activateWindow()
        self.main_interface.setFocus()

    def in_op(self, event):
        return event.inaxes == self.axes[0]

    def in_cm(self, event):
        return event.inaxes == self.axes[1]

    def onpress(self, event):
        # find closest mean - if close enough, then indicate we are updating
        if self.in_op(event):  # click inside opacity_tf
            plot_pt = np.array([event.xdata, event.ydata])
            if plot_pt[0] is None:
                return
            closest_mean_ind = np.argmin(np.linalg.norm(plot_pt - self.op_gmm_pts, axis=1))
            mean_pt = self.op_gmm_pts[closest_mean_ind, :]
            if np.linalg.norm(mean_pt - plot_pt) / self.fig_x_diam < self.selection_thresh:
                self.pt_selected = closest_mean_ind
                self.pressed_plot_pt = plot_pt
        elif self.in_cm(event):
            min_idx, _ = min(enumerate([(x[0] - event.xdata)**2 for x in self.cm_gmm_pts]), key=lambda x: x[1])
            self.pt_selected = min_idx
            self.pressed_plot_pt = np.array([event.xdata, event.ydata])

    def onmotion(self, event):
        plot_pt = np.array([event.xdata, event.ydata])
        if plot_pt[0] is None:
            return
        if self.pt_selected is None:
            self.draw()
            return
        closest_ind = np.argmin(np.absolute(plot_pt[0] - self.genren.opacity_tf[:, 0]))
        mean_pt = self.genren.opacity_tf[closest_ind, 0]
        # self.tf_cursor.set_xdata([mean_pt,mean_pt])
        # self.tf_cursor.set_ydata([0,1])

        if self.in_op(event):
            if event.button == 1:
                diff_update = plot_pt - self.pressed_plot_pt
                self.genren.opacity_gmm[self.pt_selected, 0] += diff_update[0]
                self.genren.opacity_gmm[self.pt_selected, 2] += diff_update[1]
            elif event.button == 3:
                diff_bandwidth_update = plot_pt[0] - self.pressed_plot_pt[0]
                new_bandwidth = self.genren.opacity_gmm[self.pt_selected, 1] + diff_bandwidth_update
                new_bandwidth = min(new_bandwidth, self.max_bandwidth)
                new_bandwidth = max(new_bandwidth, self.min_bandwidth)
                self.genren.opacity_gmm[self.pt_selected, 1] = new_bandwidth
        elif self.in_cm(event):
            if event.button == 1:
                print(self.pt_selected)
                diff_update = np.array([event.xdata, event.ydata]) - self.pressed_plot_pt
                diff = self.cm_to_cv(diff_update[0])
                self.genren.color_gmm[self.pt_selected, 0] += diff
            pass

        self.genren.update_gmm_transfer_function()
        self.main_interface.do_update()
        self.plot_update()
        self.pressed_plot_pt = plot_pt

    def onrelease(self, event):
        modifiers = QApplication.keyboardModifiers()
        self.modifying_modes = False
        if modifiers == QtCore.Qt.ShiftModifier:
            self.modifying_modes = True

        if self.in_op(event):
            plot_pt = np.array([event.xdata, event.ydata])
            if self.modifying_modes and plot_pt[0] is not None:
                self.modifying_modes = False
                if event.button == 1:
                    new_pt = np.array([plot_pt[0], self.min_bandwidth, plot_pt[1]])
                    self.genren.opacity_gmm = np.vstack([self.genren.opacity_gmm, new_pt])
                elif event.button == 3:
                    closest_mean_ind = np.argmin(np.linalg.norm(plot_pt - self.op_gmm_pts, axis=1))
                    mean_pt = self.op_gmm_pts[closest_mean_ind, :]
                    if np.linalg.norm(mean_pt - plot_pt) / self.fig_x_diam < self.selection_thresh:
                        subset_inds = np.arange(self.genren.opacity_gmm.shape[0]) != closest_mean_ind
                        self.genren.opacity_gmm = self.genren.opacity_gmm[subset_inds, :]
        if self.in_cm(event):
            if event.button == 3:
                color = QColorDialog().getColor()
                self.genren.color_gmm[self.pt_selected, 1:] = np.array(
                    [color.redF(), color.greenF(), color.blueF()])
                print(self.genren.color_gmm.shape)
            if self.modifying_modes and event.button == 1:
                color = QColorDialog().getColor()
                new_color = [self.cm_to_cv(event.xdata), color.redF(), color.greenF(), color.blueF()]
                cl = self.genren.color_gmm.tolist()
                cl.append(new_color)
                cl.sort(key=lambda x: x[0])
                self.genren.color_gmm = np.array(cl)

        self.genren.update_gmm_transfer_function()
        self.main_interface.do_update()
        self.plot_update()
        self.pt_selected = None
        self.pressed_plot_pt = None

    def save_fig(self, fig_filename):
        plt.savefig(fig_filename)
