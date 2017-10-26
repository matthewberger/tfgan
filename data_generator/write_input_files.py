import sys
import numpy as np
from tqdm import tqdm 

views_mat = np.load(sys.argv[1])
opacity_maps_mat = np.load(sys.argv[2])
opacity_maps_mat = opacity_maps_mat.reshape(opacity_maps_mat.shape[0],opacity_maps_mat.shape[1]*opacity_maps_mat.shape[2])
color_maps_mat = np.load(sys.argv[3])
color_maps_mat = color_maps_mat.reshape(color_maps_mat.shape[0],color_maps_mat.shape[1]*color_maps_mat.shape[2])
inputs_dir = sys.argv[4]

ind = 0
inputs_file = open(inputs_dir+'inputs.csv', 'w')
for view,opacity_map,color_map in tqdm(zip(views_mat,opacity_maps_mat,color_maps_mat)):
	input_name = 'input'+str(ind)+'.csv'
	np.savetxt(inputs_dir+input_name, np.hstack((view,opacity_map,color_map)), delimiter=',', fmt='%1.4f')
	inputs_file.write(input_name+'\n')
	ind+=1
inputs_file.close()
