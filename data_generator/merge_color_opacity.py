import sys
import numpy as np
from PIL import Image

color_filename = sys.argv[1]
opacity_filename = sys.argv[2]
merged_filename = sys.argv[3]

color_img = np.array(Image.open(color_filename),dtype=np.uint8)
opacity_img = np.array(Image.open(opacity_filename),dtype=np.uint8)
merged_img = np.zeros((color_img.shape[0],color_img.shape[1],4),dtype=np.uint8)
merged_img[:,:,:3] = color_img[:,:,:3]
merged_img[:,:,3] = opacity_img[:,:,0]
Image.fromarray(merged_img).save(merged_filename)
