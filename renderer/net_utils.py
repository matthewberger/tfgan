import numpy as np
import math
import colorsys

from colormath.color_objects import LabColor, sRGBColor, HSLColor
from colormath.color_conversions import convert_color

def normalize_opacity(opacity_func):
    return 2.0*opacity_func-1.0

def normalize_color(color_func):
    return 2.0*color_func-1.0

def normalize_rgb_to_lab(color_func):
    min_lab_l,max_lab_l = 0.0,100.0
    min_lab_a,max_lab_a = -86.185,98.254
    min_lab_b,max_lab_b = -107.863,94.482

    lab_color_func = np.zeros((color_func.shape[0],3))
    for cdx,color in enumerate(color_func):
        lab_color_func[cdx,:] = convert_color(sRGBColor(color[0],color[1],color[2]),LabColor).get_value_tuple()
    lab_color_func[:,0] = 2.0*((lab_color_func[:,0]-min_lab_l) / (max_lab_l-min_lab_l)) - 1.0
    lab_color_func[:,1] = 2.0*((lab_color_func[:,1]-min_lab_a) / (max_lab_a-min_lab_a)) - 1.0
    lab_color_func[:,2] = 2.0*((lab_color_func[:,2]-min_lab_b) / (max_lab_b-min_lab_b)) - 1.0
    return lab_color_func

def normalize_rgb_to_hsl(color_func):
    hls_color_func = np.zeros((color_func.shape[0],4))
    for cdx,color in enumerate(color_func):
        hls_color = colorsys.rgb_to_hls(color[0],color[1],color[2])
        hls_color_func[cdx,0] = np.sin(2.0*math.pi*hls_color[0])
        hls_color_func[cdx,1] = np.cos(2.0*math.pi*hls_color[0])
        hls_color_func[cdx,2] = 2.0*hls_color[1]-1.0
        hls_color_func[cdx,3] = 2.0*hls_color[2]-1.0
    return hls_color_func

def normalize_elevation(elevation,min_elevation=0.0,max_elevation=180.0):
    return 2.0*((elevation-min_elevation)/(max_elevation-min_elevation))-1.0

def normalize_azimuth(azimuth):
    return np.array([np.cos(np.radians(azimuth)),np.sin(np.radians(azimuth))])

def normalize_roll(roll):
    return roll/10.0

def normalize_zoom(zoom,min_zoom=1.0,max_zoom=2.0):
    return 2.0*((zoom-min_zoom)/(max_zoom-min_zoom))-1.0
