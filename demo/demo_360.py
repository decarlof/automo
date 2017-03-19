# Note: this applies to 360 deg samples.

import automo
import os

preview_dict_360 = {'slice_st':500, 'slice_end':501, 'slice_step':1}

automo.robo.process_folder(os.getcwd(), ini_name='automo_360.ini', preview=preview_dict_360)