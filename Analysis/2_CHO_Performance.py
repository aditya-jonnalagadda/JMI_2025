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



testdata = sio.loadmat('../VICTRE_MO-master/2017_testdata.mat')
mat_dir = '../VICTRE_MO-master/Results/' #_Gabor_with_Slice_Wts/'


CHO_MASS_2D_sig = sio.loadmat(mat_dir + 'results_2D_CHO_1_Signal.mat')
CHO_MASS_2D_noi = sio.loadmat(mat_dir + 'results_2D_CHO_1_Noise.mat')

CHO_CALC_2D_sig = sio.loadmat(mat_dir + 'results_2D_CHO_0_Signal.mat')
CHO_CALC_2D_noi = sio.loadmat(mat_dir + 'results_2D_CHO_0_Noise.mat')

CHO_MASS_3D_sig = sio.loadmat(mat_dir + 'results_3D_CHO_1_Signal.mat')
CHO_MASS_3D_noi = sio.loadmat(mat_dir + 'results_3D_CHO_1_Noise.mat')

CHO_CALC_3D_sig = sio.loadmat(mat_dir + 'results_3D_CHO_0_Signal.mat')
CHO_CALC_3D_noi = sio.loadmat(mat_dir + 'results_3D_CHO_0_Noise.mat')



# Search

CHO_Search_CALC_2D_sig = CHO_CALC_2D_sig['resp_list_search']
CHO_Search_CALC_2D_noi = CHO_CALC_2D_noi['resp_list_search']

CHO_Search_MASS_2D_sig = CHO_MASS_2D_sig['resp_list_search']
CHO_Search_MASS_2D_noi = CHO_MASS_2D_noi['resp_list_search']

CHO_Search_CALC_3D_sig = CHO_CALC_3D_sig['resp_list_3d_search']
CHO_Search_CALC_3D_noi = CHO_CALC_3D_noi['resp_list_3d_search']

CHO_Search_MASS_3D_sig = CHO_MASS_3D_sig['resp_list_3d_search']
CHO_Search_MASS_3D_noi = CHO_MASS_3D_noi['resp_list_3d_search']


# LKE

CHO_LKE_CALC_2D_sig = CHO_CALC_2D_sig['resp_list_LKE']
CHO_LKE_CALC_2D_noi = CHO_CALC_2D_noi['resp_list_LKE']

CHO_LKE_MASS_2D_sig = CHO_MASS_2D_sig['resp_list_LKE']
CHO_LKE_MASS_2D_noi = CHO_MASS_2D_noi['resp_list_LKE']

CHO_LKE_CALC_3D_sig = CHO_CALC_3D_sig['resp_list_3d']
CHO_LKE_CALC_3D_noi = CHO_CALC_3D_noi['resp_list_3d']

CHO_LKE_MASS_3D_sig = CHO_MASS_3D_sig['resp_list_3d']
CHO_LKE_MASS_3D_noi = CHO_MASS_3D_noi['resp_list_3d']



# Create a dictionary for bootstrap analysis
bs_search_dict = {}

for sig_type in ['calc', 'mass']:
  bs_search_dict[sig_type] = {}
  for dim in ['2D', '3D']:
    bs_search_dict[sig_type][dim] = {}
    for sig_noi in ['signal', 'noise']:
      bs_search_dict[sig_type][dim][sig_noi] = {}

      print('{} - {} - {}'.format(sig_type, dim, sig_noi))

      if sig_noi=='signal':
        if sig_type=='calc':
          names = testdata['calc_testnames']
          if dim=='2D':
            values = CHO_Search_CALC_2D_sig
          else:
            values = CHO_Search_CALC_3D_sig
        elif sig_type=='mass':
          names = testdata['mass_testnames']
          if dim=='2D':
            values = CHO_Search_MASS_2D_sig
          else:
            values = CHO_Search_MASS_3D_sig
      elif sig_noi=='noise':
        names = testdata['noise_testnames']
        print('names count - {}'.format(len(names)))
        if sig_type=='calc':
          if dim=='2D':
            values = CHO_Search_CALC_2D_noi
          else:
            values = CHO_Search_CALC_3D_noi
        elif sig_type=='mass':
          if dim=='2D':
            values = CHO_Search_MASS_2D_noi
          else:
            values = CHO_Search_MASS_3D_noi

      print('values: , (#names - {}, #values - {})'.format(len(names), len(values)))

      #for name, val in zip(names, values):
      #  bs_search_dict[sig_type][dim][sig_noi][name[0][0]] = val[0]
      #  print('{} - {}'.format(name[0][0], val[0]))

      current_idx = 0
      val_per_img = int(len(values)/len(names))
      for name in names:
        bs_search_dict[sig_type][dim][sig_noi][name[0][0]] = values[current_idx:(current_idx+val_per_img)]
        current_idx += val_per_img

      #print(bs_search_dict[sig_type][dim][sig_noi])
      #print(0/0)

cho_search_dict = bs_search_dict


#########################################################################

# Create a dictionary for bootstrap analysis
bs_search_dict = {}

for sig_type in ['calc', 'mass']:
  bs_search_dict[sig_type] = {}
  for dim in ['2D', '3D']:
    bs_search_dict[sig_type][dim] = {}
    for sig_noi in ['signal', 'noise']:
      bs_search_dict[sig_type][dim][sig_noi] = {}

      print('{} - {} - {}'.format(sig_type, dim, sig_noi))

      if sig_noi=='signal':
        if sig_type=='calc':
          names = testdata['calc_testnames']
          if dim=='2D':
            values = CHO_LKE_CALC_2D_sig
          else:
            values = CHO_LKE_CALC_3D_sig
        elif sig_type=='mass':
          names = testdata['mass_testnames']
          if dim=='2D':
            values = CHO_LKE_MASS_2D_sig
          else:
            values = CHO_LKE_MASS_3D_sig
      elif sig_noi=='noise':
        names = testdata['noise_testnames']
        print('names count - {}'.format(len(names)))
        if sig_type=='calc':
          if dim=='2D':
            values = CHO_LKE_CALC_2D_noi
          else:
            values = CHO_LKE_CALC_3D_noi
        elif sig_type=='mass':
          if dim=='2D':
            values = CHO_LKE_MASS_2D_noi
          else:
            values = CHO_LKE_MASS_3D_noi

      #print('values: {}, (#names - {}, #values - {})'.format(values, len(names), len(values)))

      #for name, val in zip(names, values):
      #  bs_search_dict[sig_type][dim][sig_noi][name[0][0]] = val[0]
      #  print('{} - {}'.format(name[0][0], val[0]))

      current_idx = 0
      val_per_img = int(len(values)/len(names))
      for name in names:
        bs_search_dict[sig_type][dim][sig_noi][name[0][0]] = values[current_idx:(current_idx+val_per_img)]
        current_idx += val_per_img

      #print(bs_search_dict[sig_type][dim][sig_noi])
      #print(0/0)

cho_LKE_dict = bs_search_dict




CHO_Search_CALC_2D, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_search_dict['calc']['2D']['noise'].values())), np.concatenate(list(cho_search_dict['calc']['2D']['signal'].values())))
CHO_Search_MASS_2D, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_search_dict['mass']['2D']['noise'].values())), np.concatenate(list(cho_search_dict['mass']['2D']['signal'].values())))

CHO_Search_CALC_3D, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_search_dict['calc']['3D']['noise'].values())), np.concatenate(list(cho_search_dict['calc']['3D']['signal'].values())))
CHO_Search_MASS_3D, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_search_dict['mass']['3D']['noise'].values())), np.concatenate(list(cho_search_dict['mass']['3D']['signal'].values())))

print('CHO - Search - 2D:    CALC: {:.3f}, MASS: {:.3f}'.format(CHO_Search_CALC_2D, CHO_Search_MASS_2D))
print('CHO - Search - 3D:    CALC: {:.3f}, MASS: {:.3f}'.format(CHO_Search_CALC_3D, CHO_Search_MASS_3D))




CHO_LKE_CALC_2D, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_LKE_dict['calc']['2D']['noise'].values())), np.concatenate(list(cho_LKE_dict['calc']['2D']['signal'].values())))
CHO_LKE_MASS_2D, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_LKE_dict['mass']['2D']['noise'].values())), np.concatenate(list(cho_LKE_dict['mass']['2D']['signal'].values())))

CHO_LKE_CALC_3D, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_LKE_dict['calc']['3D']['noise'].values())), np.concatenate(list(cho_LKE_dict['calc']['3D']['signal'].values())))
CHO_LKE_MASS_3D, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_LKE_dict['mass']['3D']['noise'].values())), np.concatenate(list(cho_LKE_dict['mass']['3D']['signal'].values())))

print('CHO - LKE - 2D:    CALC: {:.3f}, MASS: {:.3f}'.format(CHO_LKE_CALC_2D, CHO_LKE_MASS_2D))
print('CHO - LKE - 3D:    CALC: {:.3f}, MASS: {:.3f}'.format(CHO_LKE_CALC_3D, CHO_LKE_MASS_3D))



# Save results in a pickle file
my_dict = {}
my_dict['LKE'] = {}
my_dict['LKE']['dict'] = cho_LKE_dict
my_dict['LKE']['calc'] = {}
my_dict['LKE']['mass'] = {}
my_dict['LKE']['calc']['2D'] = CHO_LKE_CALC_2D
my_dict['LKE']['calc']['3D'] = CHO_LKE_CALC_3D
my_dict['LKE']['mass']['2D'] = CHO_LKE_MASS_2D
my_dict['LKE']['mass']['3D'] = CHO_LKE_MASS_3D
my_dict['Search'] = {}
my_dict['Search']['dict'] = cho_search_dict
my_dict['Search']['calc'] = {}
my_dict['Search']['mass'] = {}
my_dict['Search']['calc']['2D'] = CHO_Search_CALC_2D
my_dict['Search']['calc']['3D'] = CHO_Search_CALC_3D
my_dict['Search']['mass']['2D'] = CHO_Search_MASS_2D
my_dict['Search']['mass']['3D'] = CHO_Search_MASS_3D


with open('pkl_files/2_CHO.pkl', 'wb') as handle:
    pkl.dump(my_dict, handle)