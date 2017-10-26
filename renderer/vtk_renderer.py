import vtk
import numpy as np

class VTKRenderer:
    def __init__(self, volume_data, genren):
        self.volume_data = volume_data
        self.genren = genren

        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_renderer.SetBackground(0.310999694819562063,0.3400015259021897,0.4299992370489052)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetSize(256,256)
        self.renWin.AddRenderer(self.vtk_renderer)
        self.renWin.SetWindowName('VTK Renderer')
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

        # compute volume center
        volume_spacing = np.array(self.volume_data.GetSpacing())
        volume_origin = np.array(self.volume_data.GetOrigin())
        volume_dimensions = np.array(self.volume_data.GetDimensions())
        volume_max = volume_origin+volume_spacing*volume_dimensions
        volume_center = 0.5*(volume_origin+volume_max)
        volume_diag = volume_max-volume_origin

        #offset_scale = 2.25
        offset_scale = 2.0

        # camera
        self.camera = self.vtk_renderer.MakeCamera()
        self.camera.SetClippingRange(0.01,offset_scale*2.0*np.linalg.norm(volume_diag))
        self.camera.SetPosition(volume_center[0],volume_center[1],volume_center[2]+offset_scale*np.linalg.norm(volume_diag))
        self.camera.SetFocalPoint(volume_center[0],volume_center[1],volume_center[2])
        self.camera.Elevation(-85)
        self.vtk_renderer.SetActiveCamera(self.camera)

        # color map, opacity map
        self.vtk_color_map = vtk.vtkColorTransferFunction()
        for color_val in self.genren.color_tf:
            self.vtk_color_map.AddRGBPoint(color_val[0],color_val[1],color_val[2],color_val[3])
        vtk_opacity_map = vtk.vtkPiecewiseFunction()
        for op_val in self.genren.opacity_tf:
            vtk_opacity_map.AddPoint(op_val[0],op_val[1])

        # volume properties
        self.prop_volume = vtk.vtkVolumeProperty()
        self.prop_volume.ShadeOff()
        self.prop_volume.SetColor(self.vtk_color_map)
        self.prop_volume.SetScalarOpacity(vtk_opacity_map)
        self.prop_volume.SetInterpolationTypeToLinear()

        # vtk volume render
        self.mapperVolume = vtk.vtkSmartVolumeMapper()
        self.mapperVolume = vtk.vtkGPUVolumeRayCastMapper()
        self.mapperVolume.SetBlendModeToComposite()
        self.mapperVolume.SetInputData(self.volume_data)
        self.actorVolume = vtk.vtkVolume()
        self.actorVolume.SetMapper(self.mapperVolume)
        self.actorVolume.SetProperty(self.prop_volume)
        self.vtk_renderer.AddActor(self.actorVolume)

        self.iren.Initialize()

    def do_render(self, elevation, azimuth, zoom):
        # vtk tf
        self.vtk_color_map = vtk.vtkColorTransferFunction()
        for color_val in self.genren.color_tf:
            self.vtk_color_map.AddRGBPoint(color_val[0],color_val[1],color_val[2],color_val[3])
        vtk_opacity_map = vtk.vtkPiecewiseFunction()
        for op_val in self.genren.opacity_tf:
            vtk_opacity_map.AddPoint(op_val[0],op_val[1])
        self.prop_volume.SetColor(self.vtk_color_map)
        self.prop_volume.SetScalarOpacity(vtk_opacity_map)

        # vtk camera
        self.camera.Elevation(elevation)
        self.camera.Azimuth(azimuth)
        self.camera.Zoom(zoom)

        #self.
        self.iren.Render()

        # vtk undo camera
        self.camera.Zoom(1.0/zoom)
        self.camera.Azimuth(-azimuth)
        self.camera.Elevation(-elevation)
