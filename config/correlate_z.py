# -*- coding: utf-8 -*-

"""
AuTomo example to generage sample previews.
"""

from __future__ import print_function

import argparse
import os
import sys
from os.path import expanduser
import dxchange
import warnings
import numpy as np
from glob import glob
import re, shutil
from scipy.ndimage import gaussian_filter
import h5py
from scipy import ndimage

import automo.util as util

# pad zeros to the right of 1 as binary (e.g.: 1 << x, x = 3, 1 -> 1000(bin) -> 8) equv: return 2 ** x
def shift_bit_length(x):
    return 1 << (x - 1).bit_length()  # equv: return 2 ** ((x - 1).bit_length())


def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.
    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.
    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.
    Parameters
    ----------
    data : 2D ndarray
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)
    Returns
    -------
    output : 2D ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (np.fft.ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(data.shape[1] / 2)).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            np.fft.ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(data.shape[0] / 2))
    )

    return row_kernel.dot(data).dot(col_kernel)


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be
        zero if images are non-negative).
    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.
    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    src_amp : float
        The normalized average image intensity of the source image
    target_amp : float
        The normalized average image intensity of the target image
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() /\
        (src_amp * target_amp)
    return np.sqrt(np.abs(error))


def register_translation(src_image, target_image, rangeX=(None, None), rangeY=(None, None), down=0, upsample_factor=1,
                         space="real", blur=3):
    """
    Efficient subpixel image translation registration by cross-correlation.
    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.
    Parameters
    ----------
    src_image : ndarray
        Reference image.
    target_image : ndarray
        Image to register.  Must be same dimensionality as ``src_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier"
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.
    rangeX : ndarray, optional
        Boundary x-positions of the mask.
    rangeY : ndarray, optional
        Boundary y-positions of the mask.
    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).
    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    # assume complex data is already in Fourier space
    norm_max = np.max([src_image.max(), target_image.max()])
    if space.lower() == 'fourier':
        src_freq = src_image / norm_max
        target_freq = target_image / norm_max
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_image = np.array(src_image, dtype=np.complex128, copy=False) / norm_max
        target_image = np.array(target_image, dtype=np.complex128, copy=False) / norm_max
        src_freq = np.fft.fftn(src_image)
        target_freq = np.fft.fftn(target_image)
    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Crop the image size to power of 2 if rangeX and rangeY are not specified
    new_size = shift_bit_length(max(map(max, target_image.shape, src_image.shape)))
    if rangeX[0] is None:
        rangeX = [0, new_size]
    if rangeY[0] is None:
        rangeY = [0, new_size]

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj() / (np.abs(src_freq) * np.abs(target_freq))
    cross_correlation = np.fft.ifftn(image_product)
    cross_correlation = ndimage.gaussian_filter(abs(cross_correlation), sigma=blur)
    mask = np.zeros(cross_correlation.shape)
    mask[:rangeY[1] - rangeY[0] + 1, :rangeX[1] - rangeX[0] + 1] = 1
    mask = np.roll(np.roll(mask, int(rangeY[0]), axis=0), int(rangeX[0]), axis=1)

    # if rangeY[0] > 0 and rangeX[0] > 0:
    #     mask[rangeY[0]:rangeY[1], rangeX[0]:rangeX[1]] = 1
    # elif rangeY[0] < 0:
    #     mask[shape[0] + rangeY[0]:, rangeX[0]:rangeX[1]] = 1
    #     mask[:rangeY[1], rangeX[0]:rangeX[1]] = 1
    # elif rangeX[0] < 0:
    #     mask[rangeY[0]:rangeY[1], shape[1] + rangeX[0]:] = 1
    #     mask[rangeY[0]:rangeY[1], :rangeX[1]] = 1
    dxchange.write_tiff(cross_correlation, 'cs', dtype='float32')
    cross_correlation = cross_correlation * mask


    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
        target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT

    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 2)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
                              np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape),
                          dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    # We want horizontal shift to be always positive (i.e., only to the right).
    if not down:
        if shifts[1] < 0:
            shifts[1] += float(shape[1])
    else:
        if shifts[0] < 0:
            shifts[0] += float(shape[0])

    return shifts



def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Folder Prefixes")
    parser.add_argument("--shift", help="shift between datasets",default='auto')
    parser.add_argument("--new_folder",help="target folder",default='auto')
    args = parser.parse_args()

    prefix = args.prefix
    shift = args.shift
    new_folder = args.new_folder

    folder_list = sorted(glob(prefix+'*[!restack]'))

    if shift == 'auto':
        print ('Attempting to find shift automatically.')
        folder_grid = util.start_file_grid(folder_list, pattern=1)
        slice0 = None
        shift_ls = []
        for i in range(folder_grid.shape[0] - 1):
            if slice0 is None:
                slice0 = dxchange.read_tiff(os.path.join(folder_grid[i, 0],'preview_recon', 'yz_cs.tiff'))
            slice1 = dxchange.read_tiff(os.path.join(folder_grid[i+1, 0], 'preview_recon', 'yz_cs.tiff'))
            slice0 = util.equalize_histogram(slice0, bin_min=slice0.min(), bin_max=slice0.max(), n_bin=10000)
            slice1 = util.equalize_histogram(slice1, bin_min=slice1.min(), bin_max=slice1.max(), n_bin=10000)
            # slice0 = gaussian_filter(slice0, 3)
            # slice1 = gaussian_filter(slice1, 3)
            dxchange.write_tiff(slice0, 'slice0', dtype='float32')
            dxchange.write_tiff(slice1, 'slice1', dtype='float32')
            this_shift = register_translation(slice0, slice1, down=True, upsample_factor=1, rangeX=(-10, 10), rangeY=(600, 1200))
            shift_ls.append(this_shift[0])
            slice0 = np.copy(slice1)
        # shift_ls = util.most_neighbor_clustering(shift_ls, 5)
        # shift = int(np.mean(shift_ls))

    else:
        shift = int(shift) #number of slices to keep
        shift_ls = [shift] * (len(folder_list) - 1)
    
    if new_folder == 'auto':
        new_folder = prefix + '_restack'
    try:
        os.makedirs(new_folder)
    except:
        pass
    f = open(os.path.join(new_folder, 'shift.txt'), 'w')
    for shift in shift_ls:
        f.write(str(shift) + '\n')
    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])