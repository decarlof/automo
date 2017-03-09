# -*- coding: utf-8 -*-
# Utility to find center of rotation.
import tomopy
import dxchange
import numpy as np
import h5py
import os

#----------------------------------------------------------------------------------
Center = 1064
sino_st = 300
sino_end = 800
medfilt_size = 1
level = 0 # 2^level binning
ExchangeRank = 0
debug = 1
file_name = 'data.h5'
chunk_size = 50
output_path = 'recon'
output_name = os.path.join(output_path, 'recon')
#----------------------------------------------------------------------------------

sino_st /= pow(2, level)
sino_end /= pow(2, level)
Center /= pow(2, level)

chunks = []
chunk_st = sino_st
chunk_end = chunk_st + chunk_size

while chunk_end < sino_end:
    chunks.append((chunk_st, chunk_end))
    chunk_st = chunk_end
    chunk_end += chunk_size
chunks.append((chunk_st, sino_end))

for (chunk_st, chunk_end) in chunks:

    print('Chunk range: ({:d}, {:d})'.format(chunk_st, chunk_end))

    try:
        prj, flat, dark, theta = dxchange.read_aps_32id(file_name, sino=(chunk_st, chunk_end))
        print(prj.shape, flat.shape, dark.shape)
    except:
        try:
            prj, flat, dark = dxchange.read_aps_32id(file_name, sino=(chunk_st, chunk_end))
            print(prj.shape, flat.shape, dark.shape)
            f = h5py.File(file_name, "r"); dset_theta = f["/exchange/theta"]; theta = dset_theta[...]; theta = theta*np.pi/180
        except:
            f = h5py.File(file_name, "r")
            prj = f['exchange/data'][:, chunk_st:chunk_end, :].astype('float32')
            flat = f['exchange/data_white'][:, chunk_st:chunk_end, :].astype('float32')
            dark = f['exchange/data_dark'][:, chunk_st:chunk_end, :].astype('float32')
            theta = f['exchange/theta'].value.astype('float32')
            theta = theta*np.pi/180


    if debug:
        print('## Debug: after reading data:')
        print('\n** Shape of the data:'+str(np.shape(prj)))
        print('** Shape of theta:'+str(np.shape(theta)))
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

    prj = tomopy.normalize(prj, flat, dark)
    print('\n** Flat field correction done!')

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

    rec = tomopy.recon(prj, theta, center=Center, algorithm='gridrec', filter_name='parzen')
    print('\nReconstruction done!\n')

    dxchange.write_tiff_stack(rec, fname=output_name, start=chunk_st, dtype='float32')

