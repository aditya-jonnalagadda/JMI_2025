import scipy.io as sio
import numpy as np
import os

import scipy.io as sio
import numpy as np

import pickle as pkl
from sklearn.metrics import roc_auc_score, roc_curve

def compute_auc_from_CNN_responses(neg, pos):
  neg = np.array(neg)
  pos = np.array(pos)

  labels = np.concatenate((np.zeros(neg.shape[0]), np.ones(pos.shape[0])))
  feat = np.concatenate((neg, pos))
  auc = roc_auc_score(labels, feat)
  fpr, tpr, thresholds = roc_curve(labels, feat)
  #PC = np.max((1-fpr)*0.5 + tpr*0.5)
  PC = 0.5*np.mean(np.array(neg)<2.5) + 0.5*np.mean(np.array(pos)>2.5)
  #print('pos.shape: {}, neg.shape: {}'.format(pos.shape, neg.shape))
  return auc, PC


#Output_path = '../Model_Observers/August_8/'



CNN_perf = pkl.load(open('../pkl_files/Jan_CNN_perf_from_raw_ALL.pkl', 'rb'))

# Create a dictionary for bootstrap analysis
bs_search_dict = {}
for sig_type in ['calc', 'mass']:
  bs_search_dict[sig_type] = {}
  for dim in ['2D', '3D']:
    bs_search_dict[sig_type][dim] = {}

bs_search_dict['calc']['2D'] = CNN_perf['CALC_2D']
bs_search_dict['mass']['2D'] = CNN_perf['MASS_2D']
bs_search_dict['calc']['3D'] = CNN_perf['CALC_3D']
bs_search_dict['mass']['3D'] = CNN_perf['MASS_3D']

cnn_search_dict = bs_search_dict



CNN_perf = pkl.load(open('../pkl_files/Jan_CNN_perf_from_raw_cropped_eight_slices.pkl', 'rb'))

# Create a dictionary for bootstrap analysis
bs_search_dict = {}
for sig_type in ['calc', 'mass']:
  bs_search_dict[sig_type] = {}
  for dim in ['2D', '3D']:
    bs_search_dict[sig_type][dim] = {}

bs_search_dict['calc']['2D'] = CNN_perf['CALC_2D']
bs_search_dict['mass']['2D'] = CNN_perf['MASS_2D']
bs_search_dict['calc']['3D'] = CNN_perf['CALC_3D']
bs_search_dict['mass']['3D'] = CNN_perf['MASS_3D']

cnn_LKE_dict = bs_search_dict



# Search

CNN_Search_CALC_2D_sig = list(cnn_search_dict['calc']['2D']['signal'].values())
CNN_Search_CALC_2D_noi = list(cnn_search_dict['calc']['2D']['noise'].values())

CNN_Search_MASS_2D_sig = list(cnn_search_dict['mass']['2D']['signal'].values())
CNN_Search_MASS_2D_noi = list(cnn_search_dict['mass']['2D']['noise'].values())

CNN_Search_CALC_3D_sig = list(cnn_search_dict['calc']['3D']['signal'].values())
CNN_Search_CALC_3D_noi = list(cnn_search_dict['calc']['3D']['noise'].values())

CNN_Search_MASS_3D_sig = list(cnn_search_dict['mass']['3D']['signal'].values())
CNN_Search_MASS_3D_noi = list(cnn_search_dict['mass']['3D']['noise'].values())


# LKE

CNN_LKE_CALC_2D_sig = list(cnn_LKE_dict['calc']['2D']['signal'].values())
CNN_LKE_CALC_2D_noi = list(cnn_LKE_dict['calc']['2D']['noise'].values())

CNN_LKE_MASS_2D_sig = list(cnn_LKE_dict['mass']['2D']['signal'].values())
CNN_LKE_MASS_2D_noi = list(cnn_LKE_dict['mass']['2D']['noise'].values())

CNN_LKE_CALC_3D_sig = list(cnn_LKE_dict['calc']['3D']['signal'].values())
CNN_LKE_CALC_3D_noi = list(cnn_LKE_dict['calc']['3D']['noise'].values())

CNN_LKE_MASS_3D_sig = list(cnn_LKE_dict['mass']['3D']['signal'].values())
CNN_LKE_MASS_3D_noi = list(cnn_LKE_dict['mass']['3D']['noise'].values())


# Search
CNN_Search_CALC_2D, pc  = compute_auc_from_CNN_responses(CNN_Search_CALC_2D_noi, CNN_Search_CALC_2D_sig)
CNN_Search_MASS_2D, pc  = compute_auc_from_CNN_responses(CNN_Search_MASS_2D_noi, CNN_Search_MASS_2D_sig)

CNN_Search_CALC_3D, pc  = compute_auc_from_CNN_responses(CNN_Search_CALC_3D_noi, CNN_Search_CALC_3D_sig)
CNN_Search_MASS_3D, pc  = compute_auc_from_CNN_responses(CNN_Search_MASS_3D_noi, CNN_Search_MASS_3D_sig)

# LKE
CNN_LKE_CALC_2D, pc  = compute_auc_from_CNN_responses(CNN_LKE_CALC_2D_noi, CNN_LKE_CALC_2D_sig)
CNN_LKE_MASS_2D, pc  = compute_auc_from_CNN_responses(CNN_LKE_MASS_2D_noi, CNN_LKE_MASS_2D_sig)

CNN_LKE_CALC_3D, pc  = compute_auc_from_CNN_responses(CNN_LKE_CALC_3D_noi, CNN_LKE_CALC_3D_sig)
CNN_LKE_MASS_3D, pc  = compute_auc_from_CNN_responses(CNN_LKE_MASS_3D_noi, CNN_LKE_MASS_3D_sig)


print('CNN - Search - 2D:    CALC: {:.3f}, MASS: {:.3f}'.format(CNN_Search_CALC_2D, CNN_Search_MASS_2D))
print('CNN - Search - 3D:    CALC: {:.3f}, MASS: {:.3f}'.format(CNN_Search_CALC_3D, CNN_Search_MASS_3D))

print('CNN - LKE - 2D:    CALC: {:.3f}, MASS: {:.3f}'.format(CNN_LKE_CALC_2D, CNN_LKE_MASS_2D))
print('CNN - LKE - 3D:    CALC: {:.3f}, MASS: {:.3f}'.format(CNN_LKE_CALC_3D, CNN_LKE_MASS_3D))




# Save results in a pickle file
my_dict = {}
my_dict['LKE'] = {}
my_dict['LKE']['dict'] = cnn_LKE_dict
my_dict['LKE']['calc'] = {}
my_dict['LKE']['mass'] = {}
my_dict['LKE']['calc']['2D'] = CNN_LKE_CALC_2D
my_dict['LKE']['calc']['3D'] = CNN_LKE_CALC_3D
my_dict['LKE']['mass']['2D'] = CNN_LKE_MASS_2D
my_dict['LKE']['mass']['3D'] = CNN_LKE_MASS_3D
my_dict['Search'] = {}
my_dict['Search']['dict'] = cnn_search_dict
my_dict['Search']['calc'] = {}
my_dict['Search']['mass'] = {}
my_dict['Search']['calc']['2D'] = CNN_Search_CALC_2D
my_dict['Search']['calc']['3D'] = CNN_Search_CALC_3D
my_dict['Search']['mass']['2D'] = CNN_Search_MASS_2D
my_dict['Search']['mass']['3D'] = CNN_Search_MASS_3D


with open('pkl_files/4_CNN.pkl', 'wb') as handle:
    pkl.dump(my_dict, handle)