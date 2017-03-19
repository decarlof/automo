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

import h5py
import numpy 
import dxchange
import numpy as np
import tomopy

f = h5py.File('data.h5')
dset = f['exchange/data']
nang = dset.shape[0]

print('Width of peojection: {:d}'.format(dset.shape[2]))

half_ang = int(nang/2)

prj1, flt1, drk1 = dxchange.read_aps_32id('data.h5', proj=(0, 1))
prj1 = tomopy.normalize(prj1, flt1, drk1)
prj1 = -np.log(prj1)
prj1[np.isnan(prj1)] = 0
prj2, flt2, drk2 = dxchange.read_aps_32id('data.h5', proj=(half_ang, half_ang+1))
prj2 = tomopy.normalize(prj2, flt2, drk2)
prj2 = -np.log(prj2)
prj2[np.isnan(prj2)] = 0
prj3, flt3, drk3 = dxchange.read_aps_32id('data.h5', sino=(500, 501))
prj3 = tomopy.normalize(prj3, flt3, drk3)
prj3 = -np.log(prj3)
prj3[np.isnan(prj3)] = 0

dxchange.write_tiff(prj1, '000')
dxchange.write_tiff(prj2, '180')dxchange.write_tiff(np.squeeze(prj3), 'sino')
