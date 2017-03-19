# -*- coding: utf-8 -*-
# Utility to find center of rotation.
"""
Instructions for 360-degree samples:
------------------------------------
1. Run hdf5_frame_writer.py to generate projections at 0 and 180 degrees. Find center of symmetry, and derive overlap
by (width_of_projection - symm_center) * 2.
2. Open test_center_360.py. Replace the value of overlap to the number found. Modify center_st and center_end if necessary.
3. Find the best center in ./center.
4. Open rec_360.py. Supply the value of overlap. Replace the value of best_center to the center position found (without downsizing). Modify sino_start, sino_end, and level (0 = 1x, 1 = 2x, etc.) if necessary.
5. Run the script and retrieve reconstruction from ./recon.
"""

import tomopy
import dxchange
import numpy as np
import h5py

#----------------------------------------------------------------------------------
slice_no = 405

Center_st = 1030     
Center_end = 1090
overlap = 604
medfilt_size = 1
level = 0 # 2^level binning
ExchangeRank = 0
debug = 1
file_name = 'data.h5'
output_path = 'center'
#----------------------------------------------------------------------------------

def sino_360_to_180(data, overlap=0, rotation='left'):
    """
    Converts 0-360 degrees sinogram to a 0-180 sinogram.
    If the number of projections in the input data is odd, the last projection
    will be discarded.
    Parameters
    ----------
    data : ndarray
        Input 3D data.
    overlap : scalar, optional
        Overlapping number of pixels.
    rotation : string, optional
        Left if rotation center is close to the left of the
        field-of-view, right otherwise.
    Returns
    -------
    ndarray
        Output 3D data.
    """
    dx, dy, dz = data.shape

    overlap = int(np.round(overlap))

    lo = overlap//2
    ro = overlap - lo
    n = dx//2

    out = np.zeros((n, dy, 2*dz-overlap), dtype=data.dtype)

    if rotation == 'left':
        out[:, :, -(dz-lo):] = data[:n, :, lo:]
        out[:, :, :-(dz-lo)] = data[n:2*n, :, ro:][:, :, ::-1]
    elif rotation == 'right':
        out[:, :, :dz-lo] = data[:n, :, :-lo]
        out[:, :, dz-lo:] = data[n:2*n, :, :-ro][:, :, ::-1]

    return out

N_recon = Center_end - Center_st

try:
    prj, flat, dark, theta = dxchange.read_aps_32id(file_name, sino=(slice_no, slice_no+1))
except:
    prj, flat, dark = dxchange.read_aps_32id(file_name, sino=(slice_no, slice_no+1))

# Read theta from the dataset:
File = h5py.File(file_name, "r"); dset_theta = File["/exchange/theta"]; theta = dset_theta[...]; theta = theta*np.pi/180

if debug:
    print('## Debug: after reading data:')
    print('\n** Shape of the data:'+str(np.shape(prj)))
    print('** Shape of theta:'+str(np.shape(theta)))
    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

prj = tomopy.normalize(prj, flat, dark)
print('\n** Flat field correction done!')

prj = sino_360_to_180(prj, overlap=overlap, rotation='right')
print('\n** Sinogram converted!')

if debug:
    print('## Debug: after normalization:')
    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

prj = tomopy.minus_log(prj)
print('\n** minus log applied!')

if debug:
    print('## Debug: after minus log:')
    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

prj = tomopy.misc.corr.remove_neg(prj, val=0.001)
prj = tomopy.misc.corr.remove_nan(prj, val=0.001)
prj[np.where(prj == np.inf)] = 0.001

if debug:
    print('## Debug: after cleaning bad values:')
    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

prj = tomopy.remove_stripe_ti(prj,4)
print('\n** Stripe removal done!')
if debug:
    print('## Debug: after remove_stripe:')
    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

prj = tomopy.median_filter(prj,size=medfilt_size)
print('\n** Median filter done!')
if debug:
    print('## Debug: after nedian filter:')
    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))


if level>0:
    prj = tomopy.downsample(prj, level=level)
    print('\n** Down sampling done!\n')
if debug:
    print('## Debug: after down sampling:')
    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

tomopy.write_center(prj,theta,dpath=output_path,cen_range=[Center_st/pow(2,level),Center_end/pow(2,level),((Center_end - Center_st)/float(N_recon))/pow(2,level)])

