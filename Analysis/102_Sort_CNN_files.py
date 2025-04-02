import os
import scipy.io as sio

import pickle as pkl
import numpy as np

src_root = '../nnUNet/nnUNet_out/'

dest_root = 'D:/Research/2023_IEEE_TMI_Phantoms/False_Positive_Analysis/Files_Sorted/'
try:
	os.mkdir(dest_root)
except:
	pass


names_mapping = pkl.load(open('../utils/names_mapping.pkl', 'rb'))

cnn_dir = dest_root + 'CNN/'
try:
	os.mkdir(cnn_dir)
except:
	pass
for sub_dir in ['mass', 'calc']:
	try:
		os.mkdir(cnn_dir+sub_dir)
	except:
		pass

	if sub_dir=='mass':
		src_dir = src_root+'Task834_MASS3D_TEST_without_tta/'
	else:
		src_dir = src_root+'Task833_CALC3D_TEST_without_tta/'

  	#task_type = ['CALC_2D','MASS_2D','CALC_3D', 'MASS_3D']
	#criterions = [0.45, 0.8, 0.95, 0.95] #[0.5, 0.95, 0.95, 0.99] , criterions = [0.5, 0.95, 0.95, 0.99]
    #thresholds = [543, 1441, 7968, 15440]

	#for file_name in os.listdir(src_dir):
	for file_idx in names_mapping[sub_dir]['noise'].keys(): #np.arange(1,41):
		file_name = names_mapping[sub_dir]['noise'][file_idx]

		#name_split = file_name.split('_')
		#f_name = '_'.join(name_split[4:])

		dest_dir_ = cnn_dir

		
		#sub_dir = name_split[1]
		dest_dir = dest_dir_ + sub_dir + '/'


		#if name_split[2]!='absent':
		#	continue

		print([src_dir, file_name])
		#mat_contents = sio.loadmat(src_dir+file_name)
		#print([dest_dir,f_name])
		#sio.savemat(dest_dir+f_name, mat_contents)

		softmax = np.load(src_dir + 'Noise_' + str(file_idx) + '.npz')
		softmax = softmax['softmax'][1]
		#softmax = np.moveaxis(np.moveaxis(softmax['softmax'][1],-1,0),1,-1)
		print(softmax.shape)
		np.save(open(dest_dir+file_name+'.npy', 'wb'), softmax)
		#break



