import glob, os


folder_list = glob.glob('GWxxx*')
for folder in folder_list:
	os.chdir(folder)
	h5file = glob.glob('*.h5')
	command = 'python ../scripts/recon.py ' + h5file + ' center 0 1200 1 1 0 50'
	os.system(command)
	os.chdir('..')   
