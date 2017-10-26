import sys
import vtk
import numpy as np

vol_filename = sys.argv[1]

volreader = vtk.vtkXMLImageDataReader()
volreader.SetFileName(vol_filename)
volreader.Update()

vol_data = volreader.GetOutput()
vol_data.GetPointData().SetActiveAttribute('Scalars_', 0)
data_range = vol_data.GetPointData().GetScalars().GetRange()
np.save(sys.argv[2],np.array([data_range[0],data_range[1]]))

print('range:',data_range[0],data_range[1])
