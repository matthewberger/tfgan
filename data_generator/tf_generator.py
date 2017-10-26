from colormath.color_objects import LabColor, sRGBColor, HSLColor
from colormath.color_conversions import convert_color
import numpy as np


def generate_random_opacity_map(min_scalar_value, max_scalar_value, num_cps):
    opacity_mapping = []
    opacity_mapping.append((min_scalar_value, 0.0))
    prior_scalar = 0.0
    scalar_step = (max_scalar_value - min_scalar_value) / (num_cps - 1)
    for cp in range(1, num_cps - 1):
        next_max_scalar = cp * scalar_step
        rand_scalar = np.random.uniform(prior_scalar, next_max_scalar)
        opacity_scale = float(cp) / (num_cps - 1)
        rand_opacity = np.random.uniform(0, opacity_scale**2)
        opacity_mapping.append((rand_scalar, rand_opacity))
        prior_scalar = rand_scalar
    opacity_mapping.append((max_scalar_value, np.random.uniform(0, 1)))
    return np.array([[opacity[0], opacity[1]] for opacity in opacity_mapping])


def generate_random_color_map(min_scalar_value, max_scalar_value, num_cps, min_lightness=0.4, max_lightness=0.9, min_saturation=0.6, max_saturation=0.7):
    all_colors = []
    prior_lightness = None
    for cdx in range(num_cps):
        rand_hue = np.random.uniform(0.0, 360.0)
        rand_saturation = np.random.uniform(min_saturation, max_saturation)
        cur_max_lightness = (cdx + 1) * ((max_lightness - min_lightness) / num_cps)
        if prior_lightness is None:
            rand_lightness = np.random.uniform(min_lightness, cur_max_lightness)
        else:
            rand_lightness = np.random.uniform(prior_lightness, cur_max_lightness)
        prior_lightness = rand_lightness
        rand_rgb = convert_color(HSLColor(rand_hue, rand_saturation, rand_lightness), sRGBColor)
        all_colors.append(rand_rgb.get_value_tuple())
    all_colors[num_cps / 2] = (0.95, 0.95, 0.95)

    color_mapping = []
    prior_scalar_val = 0
    for rdx, rgb in enumerate(all_colors):
        next_scalar_val = min_scalar_value + (max_scalar_value - min_scalar_value) * \
            (float(rdx) / (len(all_colors) - 1))
        scalar_val = prior_scalar_val + (next_scalar_val - prior_scalar_val) * np.random.uniform(0.0, 1.0)
        if rdx == 0:
            color_mapping.append(min_scalar_value)
            prior_scalar_val = min_scalar_value
        elif rdx == len(all_colors) - 1:
            color_mapping.append(max_scalar_value)
        else:
            color_mapping.append(scalar_val)
            prior_scalar_val = scalar_val
        color_mapping.append(rgb[0])
        color_mapping.append(rgb[1])
        color_mapping.append(rgb[2])
    return np.array(color_mapping).reshape(len(color_mapping) / 4, 4)


def pw_opacity_map_sampler(min_scalar_value, max_scalar_value, opacity_mat, opacity_res, write_scalars=True):
    cur_opacity_ind = 0
    if write_scalars:
        opacity_maps = np.zeros((opacity_res, 2))
    else:
        opacity_maps = np.zeros((opacity_res, 1))

    for idx in range(opacity_res):
        interp = float(idx) / (opacity_res - 1)
        scalar_val = min_scalar_value + interp * (max_scalar_value - min_scalar_value)
        if write_scalars:
            opacity_maps[idx, 0] = scalar_val

        cur_opacity_sv = opacity_mat[cur_opacity_ind, 0]
        next_opacity_sv = opacity_mat[cur_opacity_ind + 1, 0]
        while scalar_val > next_opacity_sv:
            cur_opacity_ind += 1
            cur_opacity_sv = opacity_mat[cur_opacity_ind, 0]
            next_opacity_sv = opacity_mat[cur_opacity_ind + 1, 0]
        scalar_val_interp = (scalar_val - cur_opacity_sv) / (next_opacity_sv - cur_opacity_sv)
        if write_scalars:
            opacity_maps[idx, 1] = opacity_mat[cur_opacity_ind, 1] + scalar_val_interp * \
                (opacity_mat[cur_opacity_ind + 1, 1] - opacity_mat[cur_opacity_ind, 1])
        else:
            opacity_maps[idx, 0] = opacity_mat[cur_opacity_ind, 1] + scalar_val_interp * \
                (opacity_mat[cur_opacity_ind + 1, 1] - opacity_mat[cur_opacity_ind, 1])
    return opacity_maps


def pw_color_map_sampler(min_scalar_value, max_scalar_value, color_mat, color_res, write_scalars=True):
    cur_color_ind = 0
    if write_scalars:
        color_map = np.zeros((color_res, 4))
    else:
        color_map = np.zeros((color_res, 3))

    for idx in range(color_res):
        interp = float(idx) / (color_res - 1)
        scalar_val = min_scalar_value + interp * (max_scalar_value - min_scalar_value)
        if write_scalars:
            color_map[idx, 0] = scalar_val

        cur_color_sv = color_mat[cur_color_ind, 0]
        next_color_sv = color_mat[cur_color_ind + 1, 0]
        while scalar_val > next_color_sv:
            cur_color_ind += 1
            cur_color_sv = color_mat[cur_color_ind, 0]
            next_color_sv = color_mat[cur_color_ind + 1, 0]
        scalar_val_interp = (scalar_val - cur_color_sv) / (next_color_sv - cur_color_sv)
        if write_scalars:
            color_map[idx, 1:] = color_mat[cur_color_ind, 1:] + scalar_val_interp * \
                (color_mat[cur_color_ind + 1, 1:] - color_mat[cur_color_ind, 1:])
        else:
            color_map[idx, :] = color_mat[cur_color_ind, 1:] + scalar_val_interp * \
                (color_mat[cur_color_ind + 1, 1:] - color_mat[cur_color_ind, 1:])
    return color_map


def generate_gaussian_tf(min_scalar_value, max_scalar_value, res, write_scalars=True):
    begin_alpha = 0.1
    end_alpha = 0.9
    mean = np.random.uniform(min_scalar_value + begin_alpha * (max_scalar_value - min_scalar_value),
                             min_scalar_value + end_alpha * (max_scalar_value - min_scalar_value))
    max_stdev = min(0.5 * (mean - min_scalar_value), 0.5 * (max_scalar_value - mean))
    stdev = np.random.uniform(0.25 * max_stdev, max_stdev)
    opacity_amplitude = np.random.uniform(0.1, 1)

    peak_color = np.random.uniform(0.0, 1.0, size=3)
    min_boundary_color = np.random.uniform(0.0, 1.0, size=3)
    max_boundary_color = np.random.uniform(0.0, 1.0, size=3)
    color_mapping = np.zeros((5, 4))
    color_mapping[:, 0] = np.array([min_scalar_value, mean - 2.0 * stdev, mean, mean + 2.0 * stdev, max_scalar_value])
    color_mapping[0, 1:] = min_boundary_color
    color_mapping[1, 1:] = min_boundary_color
    color_mapping[2, 1:] = peak_color
    color_mapping[3, 1:] = max_boundary_color
    color_mapping[4, 1:] = max_boundary_color

    if write_scalars:
        opacity_map = np.zeros((res, 2))
    else:
        opacity_map = np.zeros((res, 1))

    for idx in range(res):
        interp = float(idx) / (res - 1)
        scalar_val = min_scalar_value + interp * (max_scalar_value - min_scalar_value)
        gauss_sample = opacity_amplitude * np.exp(-(scalar_val - mean)**2 / stdev**2)
        if write_scalars:
            opacity_map[idx, 0] = scalar_val
            opacity_map[idx, 1] = gauss_sample
        else:
            opacity_map[idx, 0] = gauss_sample

    color_map = pw_color_map_sampler(min_scalar_value, max_scalar_value, color_mapping, res, write_scalars)
    return opacity_map, color_map


def generate_opacity_gmm(min_scalar_value, max_scalar_value, num_modes,  min_amplitude=0.1, variance_scale=0.5, begin_alpha=0.1, end_alpha=0.9):
    means = np.random.uniform(min_scalar_value + begin_alpha * (max_scalar_value - min_scalar_value),
                              min_scalar_value + end_alpha * (max_scalar_value - min_scalar_value), size=num_modes)
    stdevs = np.zeros(num_modes)
    amplitudes = np.zeros(num_modes)
    for mdx, mean in enumerate(means):
        max_stdev = min(0.5 * (mean - min_scalar_value), 0.5 * (max_scalar_value - mean))
        stdevs[mdx] = np.random.uniform(max_stdev / num_modes, max_stdev)
        amplitudes[mdx] = np.random.uniform(min_amplitude, 1)
    if num_modes > 1:
        amplitudes /= np.sum(amplitudes)

    opacity_gmm = np.zeros((num_modes, 3))
    opacity_gmm[:, 0] = means
    opacity_gmm[:, 1] = stdevs
    opacity_gmm[:, 2] = amplitudes
    return opacity_gmm


def generate_opacity_color_gmm(min_scalar_value, max_scalar_value, num_modes, min_amplitude=0.1, variance_scale=0.5, begin_alpha=0.1, end_alpha=0.9):
    min_mean,max_mean = min_scalar_value + begin_alpha * (max_scalar_value - min_scalar_value),min_scalar_value + end_alpha * (max_scalar_value - min_scalar_value)
    means = np.random.uniform(min_mean,max_mean,size=num_modes)
    stdevs = np.zeros(num_modes)
    amplitudes = np.zeros(num_modes)

    max_boundary_lightness = 0.4
    min_mode_lightness = 0.6
    min_stdev = 0.05

    color_gmm = np.zeros((num_modes + 2, 4))
    color_gmm[0, 0] = min_scalar_value
    color_gmm[-1, 0] = max_scalar_value
    color_gmm[0, 1:] = convert_color(HSLColor(360.0 * np.random.uniform(), np.random.uniform(),
                                              np.random.uniform(0, max_boundary_lightness)), sRGBColor).get_value_tuple()
    color_gmm[-1, 1:] = convert_color(HSLColor(360.0 * np.random.uniform(), np.random.uniform(),
                                               np.random.uniform(0, max_boundary_lightness)), sRGBColor).get_value_tuple()
    for mdx, mean in enumerate(means):
        max_stdev = min(0.5 * (mean - min_mean), 0.5 * (max_mean - mean))
        lower_stdev = min(min_stdev, max_stdev/num_modes)
        stdevs[mdx] = np.random.uniform(lower_stdev, max_stdev)
        amplitudes[mdx] = np.random.uniform(min_amplitude, 1)
        color_gmm[mdx + 1, 0] = mean
        color_gmm[mdx + 1, 1:] = convert_color(HSLColor(360.0 * np.random.uniform(), np.random.uniform(),
                                                        np.random.uniform(min_mode_lightness, 1)), sRGBColor).get_value_tuple()
    sorted_color_inds = np.argsort(color_gmm[:, 0])
    color_gmm = color_gmm[sorted_color_inds, :]
    if num_modes > 1:
        amplitudes /= np.sum(amplitudes)

    # mean, stdev, amplitude
    opacity_gmm = np.zeros((num_modes, 3))
    opacity_gmm[:, 0] = means
    opacity_gmm[:, 1] = stdevs
    opacity_gmm[:, 2] = amplitudes

    sorted_opacity_inds = np.argsort(opacity_gmm[:, 0])
    sorted_opacity_gmm = opacity_gmm[sorted_opacity_inds, :]
    # smooth out colors based on GMM
    smoothed_color_gmm = np.zeros((num_modes + 2, 4))
    smoothed_color_gmm[0, :] = color_gmm[0, :]
    smoothed_color_gmm[-1, :] = color_gmm[-1, :]
    for cdx, color in enumerate(color_gmm[1:-1, :]):
        weights = np.exp(-np.power((sorted_opacity_gmm[cdx, 0] -
                                    color_gmm[:, 0]), 2) / np.power(sorted_opacity_gmm[cdx, 1], 2))
        normalized_weights = weights / np.sum(weights)
        smoothed_color_gmm[cdx + 1, 0] = sorted_opacity_gmm[cdx, 0]
        smoothed_color_gmm[cdx + 1, 1:] = np.sum((color_gmm[:, 1:].T * normalized_weights).T, axis=0)

    return opacity_gmm, smoothed_color_gmm


def generate_op_tf_from_op_gmm(opacity_gmm, min_scalar_value, max_scalar_value, res=256, write_scalars=True):
    if write_scalars:
        opacity_map = np.zeros((res, 2))
    else:
        opacity_map = np.zeros((res, 1))

    for idx in range(res):
        interp = float(idx) / (res - 1)
        scalar_val = min_scalar_value + interp * (max_scalar_value - min_scalar_value)
        gmm_sample = np.sum(opacity_gmm[:, 2] * np.exp(-np.power((scalar_val -
                                                                  opacity_gmm[:, 0]), 2) / np.power(opacity_gmm[:, 1], 2)))
        gmm_sample = min(1,gmm_sample)
        if write_scalars:
            opacity_map[idx, 0] = scalar_val
            opacity_map[idx, 1] = gmm_sample
        else:
            opacity_map[idx, 0] = gmm_sample
    return opacity_map


def generate_tf_from_gmm(opacity_gmm, color_gmm, min_scalar_value, max_scalar_value, res=256, write_scalars=True):
    opacity_map = generate_op_tf_from_op_gmm(opacity_gmm, min_scalar_value, max_scalar_value, res, write_scalars)
    if color_gmm is not None:
        color_map = pw_color_map_sampler(min_scalar_value, max_scalar_value, color_gmm, res, write_scalars)
        return opacity_map, color_map
    else:
        return opacity_map


def generate_gmm_tf(min_scalar_value, max_scalar_value, num_modes, res, min_amplitude=0.1, variance_scale=0.5, write_scalars=True):
    opacity_gmm, color_gmm = generate_opacity_color_gmm(
        min_scalar_value, max_scalar_value, num_modes, min_amplitude, variance_scale)

    if write_scalars:
        opacity_map = np.zeros((res, 2))
    else:
        opacity_map = np.zeros((res, 1))

    for idx in range(res):
        interp = float(idx) / (res - 1)
        scalar_val = min_scalar_value + interp * (max_scalar_value - min_scalar_value)
        gmm_sample = np.sum(opacity_gmm[:, 2] * np.exp(-np.power(-(scalar_val -
                                                                   opacity_gmm[:, 0]), 2) / np.power(opacity_gmm[:, 1], 2)))
        if write_scalars:
            opacity_map[idx, 0] = scalar_val
            opacity_map[idx, 1] = gmm_sample
        else:
            opacity_map[idx, 0] = gmm_sample

    color_map = pw_color_map_sampler(min_scalar_value, max_scalar_value, color_mapping, res, write_scalars)
    return opacity_map, color_map
