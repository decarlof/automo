import automo
import os

preview_dict = {'proj_st': 0, 'proj_end': 1, 'proj_step': 1, 'slice_st': 500, 'slice_end': 501, 'slice_step': 1}

center_dict = {'rot_start': 850, 'rot_end': 1000, 'rot_step': 1, 'slice': -1, 'n_slice': 5, 'medfilt_size': 3,
               'level': 0, 'padding': 1000}

recon_dict = {'center_folder': 'center', 'sino_start': 0, 'sino_end': 1199, 'sino_step': 1, 'medfilt_size': 1,
              'level': 0, 'chunk_size': 50, 'padding': 1000}

preview_dict_360 = {'slice_st': 500, 'slice_end': 501, 'slice_step': 1}

center_dict_360 = {'rot_start': 930, 'rot_end': 990, 'rot_step': 1, 'slice': -1, 'n_slice': 5, 'medfilt_size': 3,
                   'level': 0}

recon_dict_360 = {'center_folder': 'center', 'sino_start': 0, 'sino_end': 1199, 'sino_step': 1, 'medfilt_size': 1,
                  'level': 0, 'chunk_size': 50}

preview_dict_tomosaic = {'proj_st': 0, 'proj_end': 1, 'proj_step': 1, 'slice_st': 500, 'slice_end': 501,
                         'slice_step': 1, 'write_aux': True}

automo.robo.process_folder(os.getcwd(),
                           preview=preview_dict,
                           center=center_dict,
                           recon=recon_dict,
                           preview_360=preview_dict_360,
                           center_360=center_dict_360,
                           recon_360=recon_dict_360,
                           preview_tomosaic=preview_dict_tomosaic)
