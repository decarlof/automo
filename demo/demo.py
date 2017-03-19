# Note: this script is for 360 deg samples.

import automo
import os

preview_360_dict = {'slice_st':500, 'slice_end':501, 'slice_step':1}

center_dict = {'rot_start':930, 'rot_end':990, 'rot_step':1, 'slice':400, 'medfilt_size':3, 'level':0}

recon_dict = {'center_folder':'center', 'sino_start':0, 'sino_end':1199, 'sino_step':1, 'medfilt_size':1, 'level':0,
              'chunk_size':50}

automo.robo.process_folder(os.getcwd(), ini_name='automo.ini', preview=preview_360_dict, center=center_dict, recon=recon_dict)