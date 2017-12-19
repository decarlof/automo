import numpy as np
import dxchange

import argparse
import os
from glob import glob

import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("folder_name", help="name of the folder containing center files", default='center')
    parser.add_argument("--method", help="method for center search", default='auto')
    parser.add_argument("--rot_start", help="starting position of center search. Used only for vo", default='auto')
    parser.add_argument("--rot_end", help="ending position of center search. Used only for vo", default='auto')

    args = parser.parse_args()

    folder = args.folder_name
    search_method = args.method

    center_ls = []
    slice_ls = os.listdir(folder)
    slice_ls = [i for i in slice_ls if os.path.isdir(i)]
    slice_ls = map(int, slice_ls)

    for ind, i in enumerate(slice_ls):
        outpath = os.path.join(os.getcwd(), folder, str(i))

        # output center images
        if search_method == 'auto':
            try:
                center_pos = util.minimum_entropy(outpath,
                                                  mask_ratio=0.7,
                                                  ring_removal=False,
                                                  reliability_screening=True)
                print('Entropy finds center {}.'.format(center_pos))
            except:
                print('Switching to CNN...')
                center_pos = util.search_in_folder_dnn(outpath)
            if center_pos is None:
                print('Switching to CNN...')
                center_pos = util.search_in_folder_dnn(outpath)
        elif search_method == 'vo':
            h5file = glob('*.h5')
            fname = h5file[0]
            prj_shape = util.read_data_adaptive(fname, shape_only=True)
            prj, flt, drk = util.read_data_adaptive(fname,
                                                    sino=(int(prj_shape[1] / 2), int(prj_shape[1] / 2) + 1),
                                                    return_theta=False)
            prj = (prj - drk) / (flt - drk)
            prj[np.isnan(prj)] = 0
            rot_start = int(args.rot_start)
            rot_end = int(args.rot_end)
            mid = prj.shape[2] / 2
            smin = (rot_start - mid) * 2
            smax = (rot_end - mid) * 2
            center_pos = util.find_center_vo(prj, smin=smin, smax=smax, step=1)

        center_ls.append(center_pos)
        print('Center for slice: {}'.format(center_pos))
    if len(center_ls) == 1:
        center_pos = center_ls[0]
    else:
        center_pos = np.mean(util.most_neighbor_clustering(center_ls, 5), dtype='float')
    f = open('center_pos.txt', 'w')
    f.write(str(center_pos))
    f.close()