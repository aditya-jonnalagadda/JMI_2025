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


for day in ['2D']:#, '3D']:
  participants = []
  for name in os.listdir('Dataset/Radiologists/'+day):
    if len(name.split('_'))>1:
      #print(name)
      participants.append(name)
  print('participants: {} ({})'.format(participants, len(participants)))


def compute_accuracy(day, sig_type, P_idx):

  gt_arr = []
  decision_arr = []

  keys = []
  result = []

  raw_responses = []
  sig, noi = [], []

  for I_idx in range(1,29): #4): #range(11,12): #28
    #print('\nImage_idx: {}, Participant_idx: {}'.format(I_idx, P_idx))
    mat_contents = sio.loadmat('Dataset/Radiologists/'+day+'/'+participants[P_idx]+'/data'+str(I_idx)+'.mat')

    # Read the image response for this participant and image
    # dict_keys(['__header__', '__version__', '__globals__', 'stimInfo', 'resp', 'sacInfo', 'eyeMovementInfo', 't', 'et', 'timeInfo'])
    response = mat_contents['resp'][0][0]['RATING'][0][0]

    stim_name = mat_contents['stimInfo'][0][0]['imgName'][0] #mat_contents['stimInfo'][0][0][8][0]
    stim_name_ = stim_name.split('\\')
    phantom_key = stim_name_[-1]
    keys.append(phantom_key.split('-')[0])
    raw_responses.append(response)

    th_val = 2.5

    if response<=th_val:
      decision = 0
    else:
      decision = 1

    gt =  mat_contents['stimInfo'][0][0]['isSignal'][0][0]
    cue = mat_contents['stimInfo']['hfSignal'][0,0][0][0]

    keep_img = False
    if stim_name_[-2]=='NoLesion':
      f_name=stim_name_[-2]
      if (cue==0 and sig_type=='mass') or (cue==1 and sig_type=='calc'):
        keep_img = True
      print('NoLesion: {}'.format(stim_name_[-1]))
      #noi_names.append(stim_name_[-1])
      #nolist.append(stim_name_[-1])
      phantom_key_ = 'NoLesion'
    elif stim_name_[-2]=='calc_041':
      f_name=stim_name_[-2][:-4]
      if sig_type=='calc':
        keep_img = True
        print('CALC: {}'.format(stim_name_[-1]))
        #sig_names.append(stim_name_[-1])
        #yourlist.append(stim_name_[-1])
        phantom_key_ = 'calc'
    elif stim_name_[-2]=='masses_143':
      f_name=stim_name_[-2][:-6]
      if sig_type=='mass':
        keep_img = True
        print('MASS: {}'.format(stim_name_[-1]))
        #sig_names.append(stim_name_[-1])
        #mylist.append(stim_name_[-1])
        phantom_key_ = 'mass'

    if not keep_img:
      continue



    #print('stimulus name: {}, gt: {}, decision: {} ({})'.format(phantom_key, gt, decision, response))

    gt_arr.append(gt)
    decision_arr.append(decision)

    if stim_name_[-2]=='NoLesion':
      noi.append(response)
    else:
      sig.append(response)

    result.append(gt==decision)

  acc = np.mean(np.array(gt_arr)==np.array(decision_arr))
  #print('Accuracy: {}'.format(acc))
  auc, pc = compute_auc_from_CNN_responses(noi, sig)
  print('AUC: {}'.format(auc))

  return acc, keys, result, gt_arr, raw_responses, auc



calc2d_noise, mass2d_noise, calc2d_sig, mass2d_sig = [], [], [], []
calc3d_noise, mass3d_noise, calc3d_sig, mass3d_sig = [], [], [], []


# Create a dictionary for bootstrap analysis
bs_search_dict = {}

for sig_type in ['calc', 'mass']:
  bs_search_dict[sig_type] = {}
  for dim in ['2D', '3D']:
    bs_search_dict[sig_type][dim] = {}
    bs_search_dict[sig_type][dim]['signal'] = {}
    bs_search_dict[sig_type][dim]['noise'] = {}
    for sig_noi in ['signal', 'noise']:
      bs_search_dict[sig_type][dim][sig_noi] = {}

      print('\n{} - {} - {}'.format(sig_type, dim, sig_noi))

      day = dim

      # Loop over the participants
      for P_idx in range(12):
        bs_search_dict[sig_type][dim]['signal'][participants[P_idx]] = {}
        bs_search_dict[sig_type][dim]['noise'][participants[P_idx]] = {}
        print('Participant: {} ({})'.format(participants[P_idx], P_idx))
        # Loop over the images
        for I_idx in range(1,29):
          #print('\nImage_idx: {}, Participant_idx: {}'.format(I_idx, P_idx))
          mat_contents = sio.loadmat('Dataset/Radiologists/'+day+'/'+participants[P_idx]+'/data'+str(I_idx)+'.mat')

          # Read the image response for this participant and image
          # dict_keys(['__header__', '__version__', '__globals__', 'stimInfo', 'resp', 'sacInfo', 'eyeMovementInfo', 't', 'et', 'timeInfo'])
          response = mat_contents['resp'][0][0]['RATING'][0][0]

          stim_name = mat_contents['stimInfo'][0][0]['imgName'][0] #mat_contents['stimInfo'][0][0][8][0]
          stim_name_ = stim_name.split('\\')
          phantom_key = stim_name_[-1]



          cue = mat_contents['stimInfo']['hfSignal'][0,0][0][0]





          if stim_name_[-2]=='NoLesion':
            if (cue==0 and sig_type=='mass') or (cue==1 and sig_type=='calc'):
              bs_search_dict[sig_type][dim]['noise'][participants[P_idx]][phantom_key] = response
            if cue==1 and sig_type=='calc':
              if dim=='2D':
                calc2d_noise.append(phantom_key)
              else:
                calc3d_noise.append(phantom_key)
            if cue==0 and sig_type=='mass':
              if dim=='2D':
                mass2d_noise.append(phantom_key)
              else:
                mass3d_noise.append(phantom_key)
          else:
            if stim_name_[-2]=='calc_041':
              if sig_type=='calc':
                bs_search_dict['calc'][dim]['signal'][participants[P_idx]][phantom_key] = response
                if dim=='2D':
                  calc2d_sig.append(phantom_key)
                else:
                  calc3d_sig.append(phantom_key)
            else:
              if sig_type=='mass':
                bs_search_dict['mass'][dim]['signal'][participants[P_idx]][phantom_key] = response
                if dim=='2D':
                  mass2d_sig.append(phantom_key)
                else:
                  mass3d_sig.append(phantom_key)


radiologist_dict = bs_search_dict




auc_calc_2D, auc_calc_3D, auc_mass_2D, auc_mass_3D = [], [], [], []

for P_idx in range(12):

  Rad_Search_CALC_2D_sig = list(radiologist_dict['calc']['2D']['signal'][participants[P_idx]].values())
  Rad_Search_CALC_2D_noi = list(radiologist_dict['calc']['2D']['noise'][participants[P_idx]].values())

  Rad_Search_MASS_2D_sig = list(radiologist_dict['mass']['2D']['signal'][participants[P_idx]].values())
  Rad_Search_MASS_2D_noi = list(radiologist_dict['mass']['2D']['noise'][participants[P_idx]].values())

  Rad_Search_CALC_3D_sig = list(radiologist_dict['calc']['3D']['signal'][participants[P_idx]].values())
  Rad_Search_CALC_3D_noi = list(radiologist_dict['calc']['3D']['noise'][participants[P_idx]].values())

  Rad_Search_MASS_3D_sig = list(radiologist_dict['mass']['3D']['signal'][participants[P_idx]].values())
  Rad_Search_MASS_3D_noi = list(radiologist_dict['mass']['3D']['noise'][participants[P_idx]].values())


  Rad_Search_CALC_2D, pc  = compute_auc_from_CNN_responses(Rad_Search_CALC_2D_noi, Rad_Search_CALC_2D_sig)
  Rad_Search_MASS_2D, pc  = compute_auc_from_CNN_responses(Rad_Search_MASS_2D_noi, Rad_Search_MASS_2D_sig)
  Rad_Search_CALC_3D, pc  = compute_auc_from_CNN_responses(Rad_Search_CALC_3D_noi, Rad_Search_CALC_3D_sig)
  Rad_Search_MASS_3D, pc  = compute_auc_from_CNN_responses(Rad_Search_MASS_3D_noi, Rad_Search_MASS_3D_sig)
  print('cnt: CALC_2D - absent ({}) & present ({})'.format(len(Rad_Search_CALC_2D_noi), len(Rad_Search_CALC_2D_sig)))
  print('cnt: MASS_2D - absent ({}) & present ({})'.format(len(Rad_Search_MASS_2D_noi), len(Rad_Search_MASS_2D_sig)))
  print('cnt: CALC_3D - absent ({}) & present ({})'.format(len(Rad_Search_CALC_3D_noi), len(Rad_Search_CALC_3D_sig)))
  print('cnt: MASS_3D - absent ({}) & present ({})'.format(len(Rad_Search_MASS_3D_noi), len(Rad_Search_MASS_3D_sig)))

  '''

  auc, Rad_Search_CALC_2D  = compute_auc_from_CNN_responses(Rad_Search_CALC_2D_noi, Rad_Search_CALC_2D_sig)
  auc, Rad_Search_MASS_2D  = compute_auc_from_CNN_responses(Rad_Search_MASS_2D_noi, Rad_Search_MASS_2D_sig)
  auc, Rad_Search_CALC_3D  = compute_auc_from_CNN_responses(Rad_Search_CALC_3D_noi, Rad_Search_CALC_3D_sig)
  auc, Rad_Search_MASS_3D  = compute_auc_from_CNN_responses(Rad_Search_MASS_3D_noi, Rad_Search_MASS_3D_sig)
  '''
  print('Rad - Search - 2D:    CALC: {:.3f}, MASS: {:.3f}'.format(Rad_Search_CALC_2D, Rad_Search_MASS_2D))
  print('Rad - Search - 3D:    CALC: {:.3f}, MASS: {:.3f}'.format(Rad_Search_CALC_3D, Rad_Search_MASS_3D))


  auc_calc_2D.append(Rad_Search_CALC_2D)
  auc_calc_3D.append(Rad_Search_CALC_3D)
  auc_mass_2D.append(Rad_Search_MASS_2D)
  auc_mass_3D.append(Rad_Search_MASS_3D)


print('\n\nOverall Rad - Search - 2D:    CALC: {:.3f} ({:.3f}), MASS: {:.3f} ({:.3f})'.format(np.mean(auc_calc_2D), np.std(auc_calc_2D)/np.sqrt(len(auc_calc_2D)), np.mean(auc_mass_2D), np.std(auc_mass_2D)/np.sqrt(len(auc_mass_2D))))
print('Overall Rad - Search - 3D:    CALC: {:.3f} ({:.3f}), MASS: {:.3f} ({:.3f})'.format(np.mean(auc_calc_3D), np.std(auc_calc_3D)/np.sqrt(len(auc_calc_3D)), np.mean(auc_mass_3D), np.std(auc_mass_3D)/np.sqrt(len(auc_mass_3D))))





# Save results in a pickle file
my_dict = {}
my_dict['dict'] = radiologist_dict
my_dict['calc'] = {}
my_dict['mass'] = {}
my_dict['calc']['2D'] = auc_calc_2D
my_dict['calc']['3D'] = auc_calc_3D
my_dict['mass']['2D'] = auc_mass_2D
my_dict['mass']['3D'] = auc_mass_3D


with open('pkl_files/1_Radiologists.pkl', 'wb') as handle:
    pkl.dump(my_dict, handle)