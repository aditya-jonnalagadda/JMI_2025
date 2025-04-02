import os
import scipy.io as sio


src_dir = 'D:/Research/2023_IEEE_TMI_Phantoms/VICTRE_MO-master/Results/Files/'
#print(os.listdir(src_dir))

dest_root = 'D:/Research/2023_IEEE_TMI_Phantoms/False_Positive_Analysis/Files_Sorted/'
try:
	os.mkdir(dest_root)
except:
	pass

cho_dir = dest_root + 'CHO/'
fco_dir = dest_root + 'FCO/'
try:
	os.mkdir(cho_dir)
	os.mkdir(fco_dir) 
except:
	pass
for sub_dir in ['mass', 'calc']:
	try:
		os.mkdir(cho_dir+sub_dir)
		os.mkdir(fco_dir+sub_dir) 
	except:
		pass

for file_name in os.listdir(src_dir):
	name_split = file_name.split('_')
	f_name = '_'.join(name_split[4:])

	if name_split[0]=='CHO':
		dest_dir_ = cho_dir
	else:
		dest_dir_ = fco_dir

	
	sub_dir = name_split[1]
	dest_dir = dest_dir_ + sub_dir + '/'


	if name_split[2]!='absent':
		continue

	print([src_dir, file_name])
	mat_contents = sio.loadmat(src_dir+file_name)
	print([dest_dir,f_name])
	sio.savemat(dest_dir+f_name, mat_contents)


	#break



