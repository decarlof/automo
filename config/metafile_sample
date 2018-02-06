import tomosaic
import pickle
import os

prefix =
file_list = tomosaic.get_files('data_raw_1x', prefix, type='h5')
file_grid = tomosaic.start_file_grid(file_list, pattern=1)
data_format =
x_shift =
y_shift =

try:
    os.mkdir('tomosaic_misc')
except:
    pass
writer = open(os.path.join('tomosaic_misc', 'meta'), 'wb')
pickle.dump([prefix, file_grid, x_shift, y_shift], writer)
writer.close()