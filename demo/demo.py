import automo
import os

preview_dict = {'proj_st':0, 'proj_end':1, 'proj_step':1, 'slice_st':500, 'slice_end':501, 'slice_step':1}

center_dict = {'rot_start':930, 'rot_end':990, 'rot_step':1, 'slice':400, 'medfilt_size':3, 'level':0}

automo.robo.process_folder(os.getcwd(), preview=preview_dict, center=center_dict)