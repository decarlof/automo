# Note: this applies to 360 deg samples.

import automo
import os

preview_dict = {'proj_st':0, 'proj_end':1, 'proj_step':1, 'slice_st':500, 'slice_end':501, 'slice_step':1, 'write_aux':True}

automo.robo.process_folder(os.getcwd(), preview=preview_dict, robo_type='tomosaic')