"""
Author: Jixian Li <jixianli@email.arizona.edu>
Description:
    This file containes Renderer class to render volume data
    This file doesn't contain program entry point, should be used by master.py
"""

from paraview.simple import *
import vtk
from tqdm import tqdm
import os
import numpy as np


class Renderer(object):
    '''
    OSPRay Volume Renderer (multi threads)
    '''

    def __init__(self, files, outdir, args, thd_id=None):
        # initialize class variables
        self.volume_file, self.view_file, self.opacity_file, self.color_file = files
        self.base_dir, self.img_folder, self.inputs_folder = outdir
        self.scalar, self.n_samples, self.n_ambients, self.start, self.end = args
        self.thd_id = thd_id
        ##############################################################
        # setup initial states
        ##############################################################
        self.read_data()
        self.setup_render_view()
        self.setup_camera()
        self.setup_volume_renderer()
        self.setup_transfer_functions()
        self.reset_camera()
        self.setup_file_paths()

    def read_data(self):
        self.volume = XMLImageDataReader(FileName=[self.volume_file])
        self.volume.PointArrayStatus = [self.scalar]
        self.view = np.load(self.view_file)
        self.opacity = np.load(self.opacity_file)
        self.color = np.load(self.color_file)

    def setup_render_view(self):
        self.render_view = CreateRenderView('RenderView%d' % self.thd_id)
        self.render_view.ViewSize = [256, 256]
        self.render_view.Background = (0.0,
                                       0.0,
                                       0.0)

    def setup_camera(self):
        self.camera = self.render_view.GetActiveCamera()
        volume_data_reader = vtk.vtkXMLImageDataReader()
        volume_data_reader.SetFileName(self.volume_file)
        volume_data_reader.Update()
        vol = volume_data_reader.GetOutput()
        vol.GetPointData().SetActiveAttribute(self.scalar, 0)

        self.vol_spc = np.array(vol.GetSpacing())
        self.vol_ori = np.array(vol.GetOrigin())
        self.vol_dim = np.array(vol.GetDimensions())
        self.vol_max = self.vol_ori + self.vol_spc * self.vol_dim
        self.vol_cen = 0.5 * (self.vol_ori + self.vol_max)

    def setup_volume_renderer(self):
        self.display = Show(self.volume, self.render_view)
        self.display.Representation = 'Volume'
        self.display.Opacity = 0
        self.display.Shade = 1
        self.render_view.EnableOSPRay = 0
        self.display.VolumeRenderingMode = 'OSPRay Based'
        self.render_view.Shadows = 1
        self.render_view.OrientationAxesVisibility = 0
        self.render_view.SamplesPerPixel = self.n_samples
        self.render_view.AmbientSamples = self.n_ambients

    def setup_transfer_functions(self):
        ColorBy(self.display, ('POINTS', self.scalar))
        self.color_map = GetColorTransferFunction(self.scalar)
        self.opacity_map = GetOpacityTransferFunction(self.scalar)
        self.color_map.EnableOpacityMapping = 0

    def reset_camera(self):
        # TODO change this function if not using visiblemale.vti
        self.camera.SetPosition(self.vol_cen[0],
                                2 * -(self.vol_max[1] - self.vol_cen[1]),
                                self.vol_cen[2])
        self.camera.SetFocalPoint(self.vol_cen)
        self.camera.SetViewUp(0, 0, -1)
        self.camera.SetViewAngle(75)
        self.camera.SetParallelProjection(False)

    def setup_file_paths(self):
        self.file_index = open(self.base_dir + 'files_%d.csv' % self.thd_id, 'w')
        self.inputs_index = open(self.base_dir + 'inputs_%d.csv' % self.thd_id, 'w')

    def render_images(self):
        for i in tqdm(range(self.start, self.end)):
            views, opacity, colors = self.view[i, :], self.opacity[i, :], self.color[i, :]
            self.color_map.RGBPoints = [item for c in colors for item in c]
            self.opacity_map.Points = [item for (v, o) in opacity for item in (v, o, 0.5, 0)]

            elev, azimuth, roll, zoom = views
            self.camera.Elevation(elev)
            self.camera.Azimuth(azimuth)
            self.camera.Roll(roll)
            self.camera.Zoom(zoom)
            Render(self.render_view)

            # Output image
            image_name = 'vimage' + str(i) + '.png'
            SaveScreenshot(self.img_folder + image_name, viewOrLayout=self.render_view, TransparentBackground=0)

            # Log the file & index
            input_name = 'input' + str(i) + '.csv'
            flatten_opacity_map = opacity.flatten()
            flatten_color_map = colors.flatten()
            np.savetxt(self.inputs_folder + input_name,
                       np.hstack((views, flatten_opacity_map, flatten_color_map)),
                       delimiter=',', fmt='%1.4f')
            self.inputs_index.write(input_name + '\n')
            self.file_index.write(image_name + '\n')

            # reset
            # self.setup_render_view()
            # self.setup_camera()
            # self.setup_volume_renderer()
            # self.setup_transfer_functions()
            self.reset_camera()
