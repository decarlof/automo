import automo
import os

preview_dict = {'proj_start': 'auto', 'proj_end': 'auto', 'proj_step': 'auto', 'slice_start': 'auto', 'slice_end': 'auto', 'slice_step': 'auto'}

center_dict = {'rot_start': 860, 'rot_end': 1060, 'rot_step': 1, 'slice': 600, 'n_slice': -1, 'medfilt_size': 3,
               'level': 0, 'pad_length': 1000, 'debug': 0}

recon_dict = {'center_folder': 'center', 'sino_start': 0, 'sino_end': 1200, 'sino_step': 1, 'medfilt_size': 1,
              'level': 0, 'pad_length': 1000, 'chunk_size': 50, 'debug': 0}

preview_dict_360 = {'proj_start': 'auto', 'proj_end': 'auto', 'proj_step': 'auto', 'slice_start': 'auto', 'slice_end': 'auto', 'slice_step': 'auto'}

center_dict_360 = {'rot_start': 1400, 'rot_end': 1650, 'rot_step': 1, 'slice_start': 600, 'n_slice': -1,  'medfilt_size': 1,
                   'level': 0}

recon_dict_360 = {'center_folder': 'center', 'sino_start': 0, 'sino_end': 1200, 'sino_step': 1, 'medfilt_size': 1,
                  'level': 0, 'chunk_size': 50, 'pad_length': 1000, 'debug': 0}

search_center_dict = {'folder_name': 'center', 'method': 'auto', 'rot_start': 'auto', 'rot_end': 'auto'}

preview_dict_tomosaic = {'proj_st': 0, 'proj_end': 1, 'proj_step': 1, 'slice_st': 500, 'slice_end': 501,
                         'slice_step': 1, 'write_aux': True}

automo.robo.process_folder(os.getcwd(),
                           preview=preview_dict,
                           center=center_dict,
                           recon=recon_dict,
                           preview_360=preview_dict_360,
                           center_360=center_dict_360,
                           recon_360=recon_dict_360,
                           preview_tomosaic=preview_dict_tomosaic,
                           search_center=search_center_dict)
