import shutil
import glob, os
import re

shift = 1108 #number of slices to keep 
folder_list = glob.glob('Mari*') #prefix for each folder
for i, folder in enumerate(folder_list):
    file_list = glob.glob(os.path.join(os.path.join(folder, 'recon', 'recon*.tiff')))
    file_list.sort()
    if i < len(folder_list) - 1:
        for j, f in enumerate(file_list[:shift]):
            shutil.copyfile(f, os.path.join('full_stack', 'recon_{:05d}.tiff'.format(j + shift * i)))
    else:
        for j, f in enumerate(file_list):
            shutil.copyfile(f, os.path.join('full_stack', 'recon_{:05d}.tiff'.format(j + shift * i)))
