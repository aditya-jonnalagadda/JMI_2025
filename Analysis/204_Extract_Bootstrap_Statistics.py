import os
import numpy as np

import scipy.io as sio
import heapq

from scipy import stats, signal

import seaborn as sns
from matplotlib.patches import Rectangle

import pickle as pkl
from numpy import linalg as LA
#from Aug_Step_11_fixation_map_funtions import phantom_data, phantom_saccades, process_drills, process_eyemovements

import random

#save_root = 'Partial_September_tmp_Visualization_NEW_201/'
#sub_root = 'September_2023/'

#save_root = 'tmp_Visualization_NEW_201/'
#sub_root = 'Jan/'

save_root = '201_Visualization_maps/'
sub_root = 'September_2023/'

# Directory containing output files - use them for analysis
dir_root = save_root+'3D_Radiologist_data/'
#print(os.listdir(dir_root))


save_dir = save_root

def compute_norm(inp_map):
  #return LA.norm(inp_map)
  return np.sum(inp_map)


plot_dir = save_dir+'paper_plots/'
try:
  os.mkdir(plot_dir)
except:
  pass

import matplotlib.pyplot as plt







# Get the list of radiologists and the corresponding phantoms seen
def get_rads_dict(Analysis_2_dict, Analysis_3_dict, model, analysis_type, perc_val=1):
  if analysis_type==2:
    time_dict = Analysis_2_dict['3D'][str(perc_val)]['NoLesion'][model]['time_dict']
  else:
    time_dict = Analysis_3_dict['3D']['NoLesion'][model]['time_dict']
    #time_dict = Analysis_3_dict['3D']['NoLesion'][model]['time']
  #print(time_dict.keys())
  #print(time_dict)

  radiologists = time_dict.keys()
  rads_dict = {}
  for rad_name in radiologists:
    rads_dict[rad_name] = {}
  for rad_name in radiologists:
    for key in time_dict[rad_name].keys():
      value = time_dict[rad_name][key]
      if not np.isnan(value):
        rads_dict[rad_name][key] = value

  return rads_dict, radiologists #ph_list


# Get the list of radiologists and the corresponding phantoms seen
def get_ph_dict(Analysis_2_dict, Analysis_3_dict, model, analysis_type, perc_val=1):
  if analysis_type==2:
    time_dict = Analysis_2_dict['3D'][str(perc_val)]['NoLesion'][model]['time_dict']
  else:
    time_dict = Analysis_3_dict['3D']['NoLesion'][model]['time_dict']
    #time_dict = Analysis_3_dict['3D']['NoLesion'][model]['time']
  # Get the list of phantoms
  ph_list = []
  radiologists = time_dict.keys()
  for rad_name in radiologists:
    keys = []
    for key in time_dict[rad_name].keys():
      value = time_dict[rad_name][key]
      if not np.isnan(value):
        keys.append(key)
    ph_list.extend(keys)
  #print('Total samples: {}'.format(len(ph_list)))
  ph_list = list(set(ph_list))
  #print('#Phantoms: {}'.format(set(ph_list)))

  ph_dict = {}
  for ph_name in ph_list:
    ph_dict[ph_name] = {}

  for rad_name in radiologists:
    for key in time_dict[rad_name].keys():
      value = time_dict[rad_name][key]
      if not np.isnan(value):
        ph_dict[key][rad_name] = value


  return ph_dict, ph_list



def rads_and_phantoms(Analysis_2_dict, Analysis_3_dict, model, analysis_type, perc_val=1):
  rads_dict, rads_list = get_rads_dict(Analysis_2_dict, Analysis_3_dict, model, analysis_type, perc_val=perc_val)
  ph_dict, ph_list = get_ph_dict(Analysis_2_dict, Analysis_3_dict, model, analysis_type, perc_val=perc_val)

  return rads_dict, ph_dict #, rads_list, ph_list



'''
def bootstrap_on_cummulative_data(Analysis_2_dict, Analysis_3_dict, analysis_type):
  if analysis_type==2:
    root = Analysis_2_dict['3D'][str(1)]['NoLesion']
  else:
    root = Analysis_3_dict['3D']['NoLesion']

  cnn_list = root['cnn']['time']
  cho_list = root['cho']['time']
  fco_list = root['fco']['time']

  print('cnn: {} ({})'.format(cnn_list, len(cnn_list)))
  cnn_global, cho_global, fco_global = [], [], []
  cnn_cho_global, cnn_fco_global, cho_fco_global = [], [], []
  for boot_iter in range(20000):
    iter_phantoms = random.choices(np.arange(0,28), k=28)
    cnn_cho, cnn_fco, cho_fco = [], [], []
    for ph_idx in iter_phantoms:
      cnn_cho.append(cnn_list[ph_idx]-cho_list[ph_idx])
      cnn_fco.append(cnn_list[ph_idx]-fco_list[ph_idx])
      cho_fco.append(cho_list[ph_idx]-fco_list[ph_idx])

    cnn_cho_global.append(np.mean(cnn_cho))
    cnn_fco_global.append(np.mean(cnn_fco))
    cho_fco_global.append(np.mean(cho_fco))

  print('\nbootstrap_on_cummulative_data')
  print('cnn-cho: significance value: {}'.format(np.mean(np.array(cnn_cho_global)<0)))
  print('cnn-fco: significance value: {}'.format(np.mean(np.array(cnn_fco_global)<0)))
  print('cho-fco: significance value: {}'.format(np.mean(np.array(cho_fco_global)<0)))
  #choices
  return
'''


def bootstrap_on_cummulative_data_V2(Analysis_2_dict, Analysis_3_dict, analysis_type, perc_val):
  flag_original_results = False #True # No randomness

  if flag_original_results:
    bootstrap_iter = 1
  else:
    bootstrap_iter = 20000 #100000

  if analysis_type==2:
    root = Analysis_2_dict['3D'][str(perc_val)]['NoLesion']
  else:
    root = Analysis_3_dict['3D']['NoLesion']

  cnn_list = root['cnn']['time']
  cho_list = root['cho']['time']
  fco_list = root['fco']['time']

  print('cnn: ({})'.format(len(cnn_list)))
  cnn_global, cho_global, fco_global = [], [], []
  cnn_cho_global, cnn_fco_global, fco_cho_global = [], [], []
  for boot_iter in range(bootstrap_iter):
    if flag_original_results:
      iter_phantoms = np.arange(0,28)
    else:
      iter_phantoms = random.choices(np.arange(0,28), k=28)
    cnn, fco, cho = [], [], []
    for ph_idx in iter_phantoms:
      cnn.append(cnn_list[ph_idx])
      fco.append(fco_list[ph_idx])
      cho.append(cho_list[ph_idx])

    cnn_global.append(np.mean(cnn))
    fco_global.append(np.mean(fco))
    cho_global.append(np.mean(cho))

  print('\nbootstrap_on_cummulative_data_V2')
  print('Mean values - CNN: {}, CHO: {}, FCO: {}'.format(np.mean(cnn_global), np.mean(cho_global), np.mean(fco_global)))
  print('Stderr values - CNN: {}, CHO: {}, FCO: {}'.format(np.mean(cnn_global)/np.sqrt(len(cnn_global)), np.std(cho_global)/np.sqrt(len(cho_global)), np.std(fco_global)/np.sqrt(len(fco_global))))

  tmp_dict = {}

  tmp_dict['mean'] = {}
  tmp_dict['mean']['cnn'] = np.mean(cnn_global)
  tmp_dict['mean']['cho'] = np.mean(cho_global)
  tmp_dict['mean']['fco'] = np.mean(fco_global)

  tmp_dict['stderr'] = {}
  tmp_dict['stderr']['cnn'] = np.mean(cnn_global)/np.sqrt(len(cnn_global))
  tmp_dict['stderr']['cho'] = np.std(cho_global)/np.sqrt(len(cho_global))
  tmp_dict['stderr']['fco'] = np.std(fco_global)/np.sqrt(len(fco_global))

  if not flag_original_results:
    print('cnn-cho: significance value: {:.5f}'.format(np.mean( (np.array(cnn_global)-np.array(cho_global))  <0)))
    print('cnn-fco: significance value: {:.5f}'.format(np.mean( (np.array(cnn_global)-np.array(fco_global))  <0)))
    print('fco-cho: significance value: {:.5f}'.format(np.mean( (np.array(fco_global)-np.array(cho_global))  <0)))
  #choices
  return tmp_dict




analysis_type = 3
flag_rads_random_effect = True #False #True
#flag_fixed = False #True #False
perc_dict = {}
for cond_idx in [2, 3]: #range(4):
  if cond_idx==0:
    model_type = 'max'
  elif cond_idx==1:
    model_type = 'sum'
  elif cond_idx==2:
    model_type = 'calc'
  elif cond_idx==3:
    model_type = 'mass'

  print('\n\n {}'.format(model_type))

  Analysis_2_dict, Analysis_3_dict = {}, {}
  if analysis_type==2:
    Analysis_2_dict = pkl.load(open(save_dir+sub_root+model_type+'_type_2_3D_analysis.pkl', 'rb'))
    perc_values = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45]#, 50]
  else:
    Analysis_3_dict = pkl.load(open(save_dir+sub_root+model_type+'_type_3_3D_analysis.pkl', 'rb'))
    perc_values = [0]


  for p_val in perc_values:
    print('\npercentage value: {}'.format(p_val))
    cnn_rads, cnn_ph = rads_and_phantoms(Analysis_2_dict, Analysis_3_dict, 'cnn', analysis_type, perc_val=p_val)
    cho_rads, cho_ph = rads_and_phantoms(Analysis_2_dict, Analysis_3_dict, 'cho', analysis_type, perc_val=p_val)
    fco_rads, fco_ph = rads_and_phantoms(Analysis_2_dict, Analysis_3_dict, 'fco', analysis_type, perc_val=p_val)

    if flag_rads_random_effect:
      cnn_dict = cnn_rads
      cho_dict = cho_rads
      fco_dict = fco_rads
    else:
      cnn_dict = cnn_ph
      cho_dict = cho_ph
      fco_dict = fco_ph

    if False: #analysis_type==2:
      # No need to sample radiologists here, only models are involved for this analysis and each computed value is already an average across radiologists
      perc_dict[model_type] = {}
      for perc_val in [1]: #perc_values: #= 1
        print('******** Area considered - {}'.format(perc_val))
        tmp_dict = bootstrap_on_cummulative_data_V2(Analysis_2_dict, Analysis_3_dict, analysis_type, perc_val)
        perc_dict[model_type][perc_val] = tmp_dict
      continue



    #radiologists = cnn_dict.keys()

    cnn_global, cho_global, fco_global = [], [], []
    cnn_cho_global, cnn_fco_global, cho_fco_global = [], [], []
    flag_debug = False
    for boot_iter in range(20000):

      CNN_resp, CHO_resp, FCO_resp = [], [], []


      # already selected the corresponding dictionary
      # Create a list from the first level of keys and then sample/freeze from the second level of keys
      first_keys = cnn_dict.keys()
      first_choices = random.choices(list(first_keys), k=len(first_keys))

      radiologist_phantom_dict = {}
      for choice_1 in first_choices:

        # NEED TO MAKE SURE SAME RADIOLOGIST SEES SAME PHANTOM
        if choice_1 in radiologist_phantom_dict.keys():
          second_choices = radiologist_phantom_dict[choice_1]
        else:
          # Possible second keys
          second_keys = cnn_dict[choice_1]
          second_choices = random.choices(list(second_keys), k=len(second_keys))
          radiologist_phantom_dict[choice_1] = second_choices

        # Accumulate the differences
        #choice_fixed = second_choices[0]
        for choice_2 in second_choices:
          #if False: #flag_fixed:
          #  CNN_resp.append(cnn_dict[choice_1][choice_fixed])
          #  CHO_resp.append(cho_dict[choice_1][choice_fixed])
          #  FCO_resp.append(fco_dict[choice_1][choice_fixed])
          if True: #else:
            CNN_resp.append(cnn_dict[choice_1][choice_2])
            CHO_resp.append(cho_dict[choice_1][choice_2])
            FCO_resp.append(fco_dict[choice_1][choice_2])




      cnn_global.append(np.mean(CNN_resp))
      cho_global.append(np.mean(CHO_resp))
      fco_global.append(np.mean(FCO_resp))

      #print('CNN_resp: {}'.format(CNN_resp))
      cnn_cho_global.append(np.mean(np.array(CNN_resp)-np.array(CHO_resp)))
      cnn_fco_global.append(np.mean(np.array(CNN_resp)-np.array(FCO_resp)))
      cho_fco_global.append(np.mean(np.array(CHO_resp)-np.array(FCO_resp)))

    # Final results of bootstrapping
    print('Final results: \nCNN: {} ({}), \nCHO: {} ({}), \nFCO: {} ({})'.format(np.mean(cnn_global), np.std(cnn_global), np.mean(cho_global), np.std(cho_global), np.mean(fco_global), np.std(fco_global)))
    print('cnn-cho: significance value: {}'.format(np.mean(np.array(cnn_cho_global)<0)))
    print('cnn-fco: significance value: {}'.format(np.mean(np.array(cnn_fco_global)<0)))
    print('cho-fco: significance value: {}'.format(np.mean(np.array(cho_fco_global)<0)))




