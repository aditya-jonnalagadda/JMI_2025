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



for cond_idx in [2,3]: #range(4):
  if cond_idx==0:
    model_type = 'max'
  elif cond_idx==1:
    model_type = 'sum'
  elif cond_idx==2:
    model_type = 'calc'
  elif cond_idx==3:
    model_type = 'mass'



  Analysis_2_dict = pkl.load(open(save_dir+sub_root+model_type+'_type_2_3D_analysis.pkl', 'rb'))
  Analysis_3_dict = pkl.load(open(save_dir+sub_root+model_type+'_type_3_3D_analysis.pkl', 'rb'))

  perc_values = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45] #, 50]

  print('Enter: Analysis 2!!!!!')
  #fig = plt.figure(figsize=(12,6), dpi=400)
  fig = plt.figure(figsize=(6,6), dpi=200)
  q_r = 1 #2 #3
  q_c = 1 #2 #3
  x_pos = np.array([0, 0.15, 0.3])
  sig_type = 'NoLesion'
  for plot_type in ['time']: #['fix', 'time']:
    for col_idx, percn in enumerate([1]):#, 5]):
      for row_idx, flag_D in enumerate(['3D']):
        if flag_D=='2D':
          sc = 100.0 #2048*792*percn/100.0
        else:
          sc = 100.0 #2048*792*64*percn/100.0
        sc_ = sc #/10.0
        # Plot the data at (row_idx, col_idx)
        # First bar plot - fixations
        fix_mean, fix_std, time_mean, time_std = [], [], [], []
        for MO_name in ['cho', 'fco', 'cnn']:
          #time_list = Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time']
          #time_list = [sc_*z_ for z_ in time_list]
          time_list = []
          for key_1 in Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time_dict'].keys():
            for key_2 in Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time_dict'][key_1].keys():
              va = Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time_dict'][key_1][key_2]
              if not np.isnan(va):
                time_list.append(va)

          if MO_name=='cnn':
            cnn_list = time_list
          if MO_name=='cho':
            cho_list = time_list
          if MO_name=='fco':
            fco_list = time_list

          time_mean.append(np.mean(time_list))
          time_std.append(np.std(time_list)/np.sqrt(len(time_list)))
        #cho_cnn_time = stats.ttest_rel(Analysis_2_dict[flag_D][str(percn)][sig_type]['cnn']['time'], Analysis_2_dict[flag_D][str(percn)][sig_type]['cho']['time']).pvalue
        #fco_cnn_time = stats.ttest_rel(Analysis_2_dict[flag_D][str(percn)][sig_type]['cnn']['time'], Analysis_2_dict[flag_D][str(percn)][sig_type]['fco']['time']).pvalue
        #cho_fco_time = stats.ttest_rel(Analysis_2_dict[flag_D][str(percn)][sig_type]['cho']['time'], Analysis_2_dict[flag_D][str(percn)][sig_type]['fco']['time']).pvalue
        cho_cnn_time = stats.ttest_rel(cnn_list, cho_list).pvalue
        fco_cnn_time = stats.ttest_rel(cnn_list, fco_list).pvalue
        cho_fco_time = stats.ttest_rel(cho_list, fco_list).pvalue


        ax_ = plt.subplot2grid((q_r, q_c), (row_idx, col_idx))
        ax_.bar(x_pos, time_mean, width=0.08, yerr=time_std, align='center', alpha=1.0, color=['tab:green','tab:blue','tab:red'], capsize=10, label='time-spent')
        ax_.set_ylabel('Percentage of time-spent', fontsize=24)
        #ax_.set_xticks(x_pos, labels=['CHO', 'FCO', 'CNN'], fontsize=16)
        ax_.set_xticks(x_pos) #, fontsize=16)
        ax_.set_xticklabels(['CHO', 'FCO', 'CNN'], fontsize=16)
        ax_.spines['top'].set_visible(False)
        #ax_.set_title('{}: Response percentage: {} \nTime-spent: cho: {:.1e}, fco: {:.1e}, cnn: {:.1e}\ncho-cnn: {:.2e}, fco-cnn: {:.2e}, cho-fco: {:.1e}'.format(flag_D, percn, time_mean[0], time_mean[1], time_mean[2], cho_cnn_time, fco_cnn_time, cho_fco_time))
        plt.tight_layout()
        #print('Saving at: {}'.format(plot_dir+model_type+'_type_2_time_analysis.png'))
        plt.savefig(plot_dir+model_type+'_type_2_time_analysis.png')
        plt.show()


  fig = plt.figure(figsize=(12,6), dpi=400)
  q_r = 1 #2 #3
  q_c = 1 #2 #3
  x_pos = np.array([0, 0.15, 0.3])
  sig_type = 'NoLesion'
  plot_dict = {}
  for MO_name in ['cho', 'fco', 'cnn']:
    plot_dict[MO_name] = {}
    plot_dict[MO_name]['mean'] = []
    plot_dict[MO_name]['std'] = []
  for plot_type in ['time']: #['fix', 'time']:
    for col_idx, percn in enumerate(perc_values): #[1, 5, 25, 50]):#, 5]):
      for row_idx, flag_D in enumerate(['3D']):
        sc = 100.0 #2048*792*64*percn/100.0
        sc_ = sc #/10.0
        for MO_name in ['cho', 'fco', 'cnn']:
          #time_list = Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time']
          #time_list = [sc_*z_ for z_ in time_list]
          time_list = []
          for key_1 in Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time_dict'].keys():
            for key_2 in Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time_dict'][key_1].keys():
              va = Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time_dict'][key_1][key_2]
              if not np.isnan(va):
                time_list.append(va)
          plot_dict[MO_name]['mean'].append(np.mean(time_list))
          plot_dict[MO_name]['std'].append(np.std(time_list)/np.sqrt(len(time_list)))



  x_pos = perc_values #np.array([0,0.2,0.4,0.6])
  ax_ = plt.subplot2grid((q_r, q_c), (0,0))#row_idx, col_idx))
  for MO_name, MO_clr in zip(['cho', 'fco', 'cnn'], ['green','blue','red']):
    ax_.errorbar(x_pos, plot_dict[MO_name]['mean'], yerr=plot_dict[MO_name]['std'], color=MO_clr, label=MO_name.upper())
    print('MO: {}, mean: {}'.format(MO_name, plot_dict[MO_name]['mean']))
  ax_.set_ylabel('Percentage of time-spent', fontsize=24)
  ax_.set_xticks(x_pos) #, fontsize=16)
  ax_.set_xticklabels(perc_values, fontsize=16)
  #ax_.set_xticks(x_pos, labels=perc_values, fontsize=16)
  #ax_.set_yticklabels([0,20,40,60,80,100], fontsize=16)
  ax_.legend(fontsize=24)
  ax_.set_xlabel('Percentage area', fontsize=24)
  ax_.spines['top'].set_visible(False)
  plt.tight_layout()
  print('Saving at: {}'.format(plot_dir+model_type+'_type_2_time_analysis.png'))
  plt.savefig(plot_dir+model_type+'_type_2_time_analysis_NEW.png')
  plt.show()





  # Plotting Analysis_3
  #print(Analysis_3_dict)
  # Plotting Analysis-3
  print('Enter: Analysis 3!!!!!')
  fig = plt.figure(figsize=(6,6), dpi=400)
  q_r = 1 #2 #1 #3
  q_c = 1 #2
  x_pos = np.array([0, 0.15, 0.3, 0.45])
  for row_idx, percn in enumerate(['3D']):
    for _, sig_type in enumerate(['NoLesion']): #, 'calc', 'mass']):
      # Plot the data at (row_idx, col_idx)
      # First bar plot - fixations
      fix_mean, fix_std, time_mean, time_std = [], [], [], []
      for MO_name in ['cho', 'fco', 'cnn']:
        time_list = [] #Analysis_3_dict[str(percn)][sig_type][MO_name]['time']
        for key_1 in Analysis_3_dict[str(percn)][sig_type][MO_name]['time_dict'].keys():
          for key_2 in Analysis_3_dict[str(percn)][sig_type][MO_name]['time_dict'][key_1].keys():
            va = Analysis_3_dict[str(percn)][sig_type][MO_name]['time_dict'][key_1][key_2]
            if not np.isnan(va):
              time_list.append(va)
        time_mean.append(np.mean(time_list))
        time_std.append(np.std(time_list)/np.sqrt(len(time_list)))
        if MO_name=='cnn':
          time_list = []
          print('\n\ntime_time')
          for key_1 in Analysis_3_dict['3D']['NoLesion']['cnn']['time_time'].keys():
            #print(x)
            for key_2 in Analysis_3_dict['3D']['NoLesion']['cnn']['time_time'][key_1].keys():
              va = Analysis_3_dict['3D']['NoLesion']['cnn']['time_time'][key_1][key_2]
              print(va)
              if not np.isnan(va):
                time_list.append(va)
          print('time_list length: {}'.format(len(time_list)))
          time_mean.append(np.mean(time_list))
          time_std.append(np.std(time_list)/np.sqrt(len(time_list)))

      print('************time_mean: {}'.format(time_mean))
      '''
      cho_cnn_time = stats.ttest_rel(Analysis_3_dict[str(percn)][sig_type]['cnn']['time'], Analysis_3_dict[str(percn)][sig_type]['cho']['time']).pvalue
      print('\n cnn-cho-fco-time: cnn: {}, cho: {}, fco: {}'.format(Analysis_3_dict[str(percn)][sig_type]['cnn']['time'], Analysis_3_dict[str(percn)][sig_type]['cho']['time'], Analysis_3_dict[str(percn)][sig_type]['fco']['time']))
      fco_cnn_time = stats.ttest_rel(Analysis_3_dict[str(percn)][sig_type]['cnn']['time'], Analysis_3_dict[str(percn)][sig_type]['fco']['time']).pvalue
      cho_fco_time = stats.ttest_rel(Analysis_3_dict[str(percn)][sig_type]['cho']['time'], Analysis_3_dict[str(percn)][sig_type]['fco']['time']).pvalue
      print('cho_cnn_time: {}, fco_cnn_time: {}'.format(cho_cnn_time, fco_cnn_time))
      '''
      print('({},{}): time_mean: {}'.format(row_idx, col_idx, time_mean))


      ax_ = plt.subplot2grid((q_r, q_c), (row_idx, 0))
      ax_.bar(x_pos, time_mean, width=0.1, yerr=time_std, align='center', alpha=1.0, color=['tab:green','tab:blue','tab:red', 'tab:grey'], capsize=10, label='time-spent')
      #ax_.set_xticks(x_pos, labels=['CHO-\nHuman', 'FCO-\nHuman', 'CNN-\nHuman', 'Human-\nHuman'], fontsize=16)
      ax_.set_xticks(x_pos) #, fontsize=16)
      ax_.set_xticklabels(['CHO-\nHuman', 'FCO-\nHuman', 'CNN-\nHuman', 'Human-\nHuman'], fontsize=16)
      ax_.set_ylabel('Pearson Correlation coefficient', fontsize=20)
      ax_.spines['top'].set_visible(False)
      #ax_.set_title('{}: Signal type: {}, \nTime-spent: cho: {:.1e}, fco: {:.1e}, cnn: {:.1e}\ncho-cnn: {:.2e}, fco-cnn: {:.2e}, cho-fco: {:.1e}'.format(percn, sig_type, time_mean[0], time_mean[1], time_mean[2], cho_cnn_time, fco_cnn_time, cho_fco_time))
      #print('{}: Signal type: {}, \nTime-spent: cho: {:.1e}, fco: {:.1e}, cnn: {:.1e}\ncho-cnn: {:.2e}, fco-cnn: {:.2e}, cho-fco: {:.1e}'.format(percn, sig_type, time_mean[0], time_mean[1], time_mean[2], cho_cnn_time, fco_cnn_time, cho_fco_time))


      #ax.set_title('{}: Signal type: {}, \nFixations: cho: {:.1e}, fco: {:.1e}, cnn: {:.1e}\ncho-cnn: {:.1e}, fco-cnn: {:.1e}, cho-fco: {:.1e}\nTime-spent: cho: {:.1e}, fco: {:.1e}, cnn: {:.1e}\ncho-cnn: {:.2e}, fco-cnn: {:.2e}, cho-fco: {:.1e}'.format(percn, sig_type, fix_mean[0], fix_mean[1], fix_mean[2], cho_cnn_fix, fco_cnn_fix, cho_fco_fix, time_mean[0], time_mean[1], time_mean[2], cho_cnn_time, fco_cnn_time, cho_fco_time))
  plt.tight_layout()
  plt.savefig(plot_dir+model_type+'_type_3_2D_3D_analysis.png')
  plt.show()


