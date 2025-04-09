# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import numpy as np
from skimage.util import view_as_windows
from scipy.ndimage import gaussian_filter


def create_apodization(window_size, pad_int=None):
    if not pad_int:
        pad_int = int(np.mean(window_size) / 16)
    yr = window_size[0] - int(window_size[0] / pad_int) * 2
    xr = window_size[1] - int(window_size[1] / pad_int) * 2
    padx = int((window_size[0] - xr) / 2)
    pady = int((window_size[1] - yr) / 2)
    pad = ((padx, padx), (pady, pady))

    weight = np.ones((xr, yr))
    weight = np.pad(weight, pad, mode="constant")
    weight = gaussian_filter(weight, np.mean(pad) / 2)
    weight[weight < 1e-14] = 1e-14
    return weight


def blocking(arr, block_shape, step):
    """
    Create a stacked array, with thumbnail images of size block_shape,
    which have an overlap specified by step. (if step = block_shape
    there is no overlap)
    """
    view = view_as_windows(arr, block_shape, step=step)
    vx = view.shape[0]
    vy = view.shape[1]
    flatten_view = view.reshape(-1,
                                view.shape[-3], view.shape[-2], view.shape[-1])
    return flatten_view, vx, vy


def inverse_blocking(block_arr, out_size, winsize, weight, step, vx, vy):
    """
    Given a stacked array of image thumbnails, expand back to unstacked
    image of size out_size, given a step size and winsize.
    Uses the inverse of weight to coadd the thumbnail images
    """
    vview = block_arr.reshape((vx, vy) + block_arr.shape[1:])
    out = np.zeros(out_size + (block_arr.shape[-1],))
    weights = np.zeros(out_size) + 1e-14
    w2 = int(winsize / 2)
    for i in range(vview.shape[0]):
        if i == 0:
            ci = w2
        else:
            ci = w2 + i * step
        for j in range(vview.shape[1]):
            if j == 0:
                cj = w2
            else:
                cj = w2 + j * step
            exist = out[ci - w2: ci + w2, cj - w2: cj + w2, :]
            new = np.einsum("ijk,ij->ijk", vview[i, j], weight)
            out[ci - w2: ci + w2, cj - w2: cj + w2, :] = new + exist
            wexist = weights[ci - w2: ci + w2, cj - w2: cj + w2]
            weights[ci - w2: ci + w2, cj - w2: cj + w2] = weight + wexist
    final = np.einsum("ijk,ij->ijk", out, 1 / weights)
    return final, weights
