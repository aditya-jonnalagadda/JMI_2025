import matplotlib.pyplot as plt
import pickle as pkl

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from prettytable import PrettyTable


dict_root = 'pkl_files/'
out_root = 'output_plots/'


# Load dictionaries
rad_dict = pkl.load(open(dict_root+'1_Radiologists.pkl', 'rb'))
cho_dict = pkl.load(open(dict_root+'2_CHO.pkl', 'rb'))
fco_dict = pkl.load(open(dict_root+'3_FCO.pkl', 'rb'))
cnn_dict = pkl.load(open(dict_root+'4_CNN.pkl', 'rb'))



# LKE Plot
fig, (ax2d, ax3d) = plt.subplots(nrows=2, figsize=(8,8), dpi=300, constrained_layout=True)

labels = ['CALC\nLKE', 'MASS\nLKE', 'CALC\nSearch', 'MASS\nSearch']


#CNN_2d = [0.964, 0.964, 0.959, 0.954]
#CHO_2d = [1.0, 0.999, 0.762, 0.810]
#FCO_2d = [1.0, 1.0, 0.739, 0.879]

#CNN_3d = [1.0, 1.0, 1.0, 0.959]
#CHO_3d = [1.0, 0.999, 0.839, 0.735]
#FCO_3d = [1.0, 1.0, 0.885, 0.747]


CNN_2d = [cnn_dict['LKE']['calc']['2D'], cnn_dict['LKE']['mass']['2D'], cnn_dict['Search']['calc']['2D'], cnn_dict['Search']['mass']['2D']]
CHO_2d = [cho_dict['LKE']['calc']['2D'], cho_dict['LKE']['mass']['2D'], cho_dict['Search']['calc']['2D'], cho_dict['Search']['mass']['2D']]
FCO_2d = [fco_dict['LKE']['calc']['2D'], fco_dict['LKE']['mass']['2D'], fco_dict['Search']['calc']['2D'], fco_dict['Search']['mass']['2D']]

CNN_3d = [cnn_dict['LKE']['calc']['3D'], cnn_dict['LKE']['mass']['3D'], cnn_dict['Search']['calc']['3D'], cnn_dict['Search']['mass']['3D']]
CHO_3d = [cho_dict['LKE']['calc']['3D'], cho_dict['LKE']['mass']['3D'], cho_dict['Search']['calc']['3D'], cho_dict['Search']['mass']['3D']]
FCO_3d = [fco_dict['LKE']['calc']['3D'], fco_dict['LKE']['mass']['3D'], fco_dict['Search']['calc']['3D'], fco_dict['Search']['mass']['3D']]


x = np.arange(len(labels))  # the label locations
width = 0.08  # the width of the bars
bar_width = 0.14


ax2d.bar(x-2*width, CHO_2d, bar_width, label='CHO', color='tab:green')
ax2d.bar(x, FCO_2d, bar_width, label='FCO', color='tab:blue')
ax2d.bar(x+2*width, CNN_2d, bar_width, label='CNN', color='tab:red')

ax2d.set_ylabel('AUC', fontsize=18)
ax2d.set_xticks(x)
ax2d.set_xticklabels(labels, fontsize=14)
ax2d.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=14)
ax2d.legend()
ax2d.set_ylim(ymin=0.5, ymax=1.05)
ax2d.axhline(y=1.0, color='slategray', linestyle='--')

ax2d.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left') #fig.tight_layout()
ax2d.set_title('2D', fontsize=16)




ax3d.bar(x-2*width, CHO_3d, bar_width, label='CHO', color='tab:green')
ax3d.bar(x, FCO_3d, bar_width, label='FCO', color='tab:blue')
ax3d.bar(x+2*width, CNN_3d, bar_width, label='CNN', color='tab:red')

ax3d.set_ylabel('AUC', fontsize=18)
ax3d.set_xticks(x)
ax3d.set_xticklabels(labels, fontsize=14)
ax3d.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=14)
ax3d.legend()
ax3d.set_ylim(ymin=0.5, ymax=1.05)
ax3d.axhline(y=1.0, color='slategray', linestyle='--')

ax3d.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left') #fig.tight_layout()
ax3d.set_title('3D', fontsize=16)

plt.savefig(out_root+'5_LKE_plot.png')








# Search plot

fig, ax2 = plt.subplots(ncols=1, figsize=(8,4), dpi=300, constrained_layout=True)

labels = ['CALC 2D', 'MASS 2D', 'CALC 3D', 'MASS 3D']


#Radiologist_mean = [0.98, 0.786, 0.864, 0.871]
#Radiologist_stderr = [0.009, 0.045, 0.032, 0.032]

#CNN = [0.959, 1.0, 0.954, 0.959]
#CHO = [0.762, 0.839, 0.810, 0.735]
#FCO = [0.739, 0.885, 0.879, 0.747]

rad_calc_2D_AUC = rad_dict['calc']['2D']
rad_calc_3D_AUC = rad_dict['calc']['3D']
rad_mass_2D_AUC = rad_dict['mass']['2D']
rad_mass_3D_AUC = rad_dict['mass']['3D']
Radiologist_mean = [np.mean(rad_calc_2D_AUC), np.mean(rad_mass_2D_AUC), np.mean(rad_calc_3D_AUC), np.mean(rad_mass_3D_AUC)]
Radiologist_stderr = [np.std(rad_calc_2D_AUC)/np.sqrt(len(rad_calc_2D_AUC)), np.std(rad_mass_2D_AUC)/np.sqrt(len(rad_mass_2D_AUC)), np.std(rad_calc_3D_AUC)/np.sqrt(len(rad_calc_3D_AUC)), np.std(rad_mass_3D_AUC)/np.sqrt(len(rad_mass_3D_AUC))]
CNN = [cnn_dict['Search']['calc']['2D'], cnn_dict['Search']['mass']['2D'], cnn_dict['Search']['calc']['3D'], cnn_dict['Search']['mass']['3D']]
CHO = [cho_dict['Search']['calc']['2D'], cho_dict['Search']['mass']['2D'], cho_dict['Search']['calc']['3D'], cho_dict['Search']['mass']['3D']]
FCO = [fco_dict['Search']['calc']['2D'], fco_dict['Search']['mass']['2D'], fco_dict['Search']['calc']['3D'], fco_dict['Search']['mass']['3D']]


x = np.arange(len(labels))  # the label locations
width = 0.08  # the width of the bars
bar_width = 0.14


ax2.bar(x-3*width, CHO, bar_width, label='CHO', color='tab:green')
ax2.bar(x-width, FCO, bar_width, label='FCO', color='tab:blue')
ax2.bar(x+width, CNN, bar_width, label='CNN', color='tab:red')
ax2.bar(x+3*width, Radiologist_mean, bar_width, yerr=Radiologist_stderr, label='Radiologist', color='tab:gray')


ax2.set_ylabel('AUC', fontsize=18)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=14)
ax2.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=14)
ax2.legend()
ax2.set_ylim(ymin=0.5, ymax=1.05)

ax2.axhline(y=1.0, color='slategray', linestyle='--')
#ax2.axhline(y=0.5, color='slategray', linestyle='--')

ax2.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left') #fig.tight_layout()
#ax1.legend()
#ax2.legend()
#plt.title('Performance on Test set', fontsize=18)

plt.savefig(out_root+'5_Search_plot.png')


out_table = PrettyTable()
out_table.field_names = ["Model observer", "calc-2d", "mass-2d", "calc-3d", "mass-3d"]
out_table.add_row(['Radiologists',  int(100*Radiologist_mean[0])/100, int(100*Radiologist_mean[1])/100, int(100*Radiologist_mean[2])/100, int(100*Radiologist_mean[3])/100])
out_table.add_row(['CNN',  int(100*CNN[0])/100, int(100*CNN[1])/100, int(100*CNN[2])/100, int(100*CNN[3])/100])
out_table.add_row(['CHO',  int(100*CHO[0])/100, int(100*CHO[1])/100, int(100*CHO[2])/100, int(100*CHO[3])/100])
out_table.add_row(['FCO', int(100*FCO[0])/100, int(100*FCO[1])/100, int(100*FCO[2])/100, int(100*FCO[3])/100])
#out_table.add_row(['--------', '--------', '--------', '--------',  '--------'])




print(out_table)

#################################################################################################################################################################
#################################################################################################################################################################




import random


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




radiologist_dict = rad_dict['dict']
cho_search_dict = cho_dict['Search']['dict']
fco_search_dict = fco_dict['Search']['dict']
cnn_search_dict = cnn_dict['Search']['dict']


out_table = PrettyTable()
out_table.field_names = ["Signal type", "dim", "Model observer", "Mean", "Left - 2.5%", "Right - 2.5%", "Left - 5%", "Right - 5%"]



flag_search = True
flag_original_results = False #True # No randomness

if flag_original_results:
  bootstrap_iter = 1
else:
  bootstrap_iter = 10000

signal_types = ['calc', 'mass'] # ['mass'] #['calc', 'mass']
dimens = ['2D', '3D'] # ['3D'] #['2D', '3D']

AUC_dict = {}
for sig_type in signal_types: #['mass']: #['calc', 'mass']:
  AUC_dict[sig_type] = {}
  for dim in dimens: #['3D']: # ['2D', '3D']:
    AUC_dict[sig_type][dim] = {}
    AUC_dict[sig_type][dim]['Rad'], AUC_dict[sig_type][dim]['CNN'], AUC_dict[sig_type][dim]['CHO'], AUC_dict[sig_type][dim]['FCO'] = [], [], [], []


    for boot_iter in range(bootstrap_iter):

      if boot_iter%1000==0:
        print('{}/{}'.format(boot_iter, bootstrap_iter))
      rad_values_dict, cnn_values_dict, cho_values_dict, fco_values_dict = {}, {}, {}, {}

      for sig_noi in ['signal', 'noise']:
        rad_values_dict[sig_noi], cnn_values_dict[sig_noi], cho_values_dict[sig_noi], fco_values_dict[sig_noi] = {}, {}, {}, {}

      for sig_noi in ['signal', 'noise']:
        first_keys = radiologist_dict[sig_type][dim][sig_noi].keys()

        if flag_original_results:
          first_choices = first_keys
        else:
          if sig_noi == 'signal':
            first_choices = random.choices(list(first_keys), k=len(first_keys))
            first_choices_sig = first_choices
          else:
            first_choices = first_choices_sig
            first_choices_sig = []

        radiologist_phantom_dict = {}
        for choice_1 in first_choices:
          #************* USE SAME RADIOLOGISTS FOR SIGNAL AND NOISE
          rad_values_dict[sig_noi][choice_1], cnn_values_dict[sig_noi][choice_1], cho_values_dict[sig_noi][choice_1], fco_values_dict[sig_noi][choice_1] = [], [], [], []

          # NEED TO MAKE SURE SAME RADIOLOGIST SEES SAME PHANTOM
          if choice_1 in radiologist_phantom_dict.keys():
            second_choices_reg = radiologist_phantom_dict[choice_1]['reg']
            second_choices_mo  = radiologist_phantom_dict[choice_1]['mo']
          else:
            # Possible second keys
            second_keys_reg = radiologist_dict[sig_type][dim][sig_noi][choice_1].keys()
            second_keys_mo = cho_search_dict[sig_type][dim][sig_noi].keys()
            if flag_original_results:
              second_choices = [] #second_keys
            else:
              second_choices_reg = random.choices(list(second_keys_reg), k=len(second_keys_reg))
              second_choices_mo = random.choices(list(second_keys_mo), k=len(second_keys_reg))
            radiologist_phantom_dict[choice_1] = {}
            radiologist_phantom_dict[choice_1]['reg'] = second_choices_reg
            radiologist_phantom_dict[choice_1]['mo'] = second_choices_mo

          for choice_2_reg, choice_2_mo in zip(second_choices_reg, second_choices_mo):
            rad_values_dict[sig_noi][choice_1].append(radiologist_dict[sig_type][dim][sig_noi][choice_1][choice_2_reg])
            if flag_search:
              cnn_values_dict[sig_noi][choice_1].append(cnn_search_dict[sig_type][dim][sig_noi][choice_2_mo])
              cho_values_dict[sig_noi][choice_1].append(cho_search_dict[sig_type][dim][sig_noi][choice_2_mo])
              fco_values_dict[sig_noi][choice_1].append(fco_search_dict[sig_type][dim][sig_noi][choice_2_mo])
            else:
              cnn_values_dict[sig_noi][choice_1].append(cnn_LKE_dict[sig_type][dim][sig_noi][choice_2_mo])
              cho_values_dict[sig_noi][choice_1].append(cho_LKE_dict[sig_type][dim][sig_noi][choice_2_mo])
              fco_values_dict[sig_noi][choice_1].append(fco_LKE_dict[sig_type][dim][sig_noi][choice_2_mo])


      #Rad_AUC, pc  = compute_auc_from_CNN_responses(rad_values['noise'], rad_values['signal'])
      #CNN_AUC, pc  = compute_auc_from_CNN_responses(cnn_values['noise'], cnn_values['signal'])
      #CHO_AUC, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_values['noise'])), np.concatenate(list(cho_values['signal'])))
      #FCO_AUC, pc  = compute_auc_from_CNN_responses(np.concatenate(list(fco_values['noise'])), np.concatenate(list(fco_values['signal'])))

      rad_auc, cnn_auc, cho_auc, fco_auc = [], [], [], []
      for choice_1 in first_choices:
        Rad_AUC_, pc  = compute_auc_from_CNN_responses(rad_values_dict['noise'][choice_1], rad_values_dict['signal'][choice_1])
        CNN_AUC_, pc  = compute_auc_from_CNN_responses(cnn_values_dict['noise'][choice_1], cnn_values_dict['signal'][choice_1])
        CHO_AUC_, pc  = compute_auc_from_CNN_responses(np.concatenate(list(cho_values_dict['noise'][choice_1])), np.concatenate(list(cho_values_dict['signal'][choice_1])))
        FCO_AUC_, pc  = compute_auc_from_CNN_responses(np.concatenate(list(fco_values_dict['noise'][choice_1])), np.concatenate(list(fco_values_dict['signal'][choice_1])))
        rad_auc.append(Rad_AUC_)
        cnn_auc.append(CNN_AUC_)
        cho_auc.append(CHO_AUC_)
        fco_auc.append(FCO_AUC_)
        #print('FCO - {} - AUC - {}, Noise - {}, Signal - {}'.format(choice_1, FCO_AUC_, np.squeeze(np.concatenate(list(fco_values_dict['noise'][choice_1]))), np.squeeze(np.concatenate(list(fco_values_dict['signal'][choice_1])))))
      Rad_AUC = np.mean(rad_auc)
      CNN_AUC = np.mean(cnn_auc)
      CHO_AUC = np.mean(cho_auc)
      FCO_AUC = np.mean(fco_auc)

      AUC_dict[sig_type][dim]['Rad'].append(Rad_AUC)
      AUC_dict[sig_type][dim]['CNN'].append(CNN_AUC)
      AUC_dict[sig_type][dim]['CHO'].append(CHO_AUC)
      AUC_dict[sig_type][dim]['FCO'].append(FCO_AUC)


for sig_type in signal_types: #['calc', 'mass']:

  for dim in dimens: #['2D', '3D']:
    print('\n{} - {}'.format(sig_type, dim))
    out_table.add_row([sig_type, dim, '', '',  '', '',  '', ''])
    for MO in ['Rad', 'CNN', 'CHO', 'FCO']:
      print('After Bootstrap - {} : Mean {:.3f} - Interval bounds ({:.3f} - {:.3f})'.format(MO, np.mean(AUC_dict[sig_type][dim][MO]), np.percentile(AUC_dict[sig_type][dim][MO], 2.5), np.percentile(AUC_dict[sig_type][dim][MO], 97.5)))
      M_1 = int(100*np.mean(AUC_dict[sig_type][dim][MO]))/100
      M_2 = int(100*np.percentile(AUC_dict[sig_type][dim][MO], 2.5))/100
      M_3 = int(100*np.percentile(AUC_dict[sig_type][dim][MO], 97.5))/100
      M_4 = int(100*np.percentile(AUC_dict[sig_type][dim][MO], 5))/100
      M_5 = int(100*np.percentile(AUC_dict[sig_type][dim][MO], 95))/100
      out_table.add_row(['',  '', MO, M_1, M_2, M_3, M_4, M_5])

    out_table.add_row(['--------', '--------', '--------', '--------',  '--------', '--------',  '--------', '--------'])
    if not flag_original_results:
      # Compute the significance values : Hypothesis for search task -  CNN > radiologist > FCO > CHO
      # wrt CNN
      CNN_Rad = np.array(AUC_dict[sig_type][dim]['CNN']) - np.array(AUC_dict[sig_type][dim]['Rad'])
      CNN_CHO = np.array(AUC_dict[sig_type][dim]['CNN']) - np.array(AUC_dict[sig_type][dim]['CHO'])
      CNN_FCO = np.array(AUC_dict[sig_type][dim]['CNN']) - np.array(AUC_dict[sig_type][dim]['FCO'])
      print('p value - CNN-Rad : {}'.format(np.mean(CNN_Rad<0)))
      print('p value - CNN-CHO : {}'.format(np.mean(CNN_CHO<0)))
      print('p value - CNN-FCO : {}'.format(np.mean(CNN_FCO<0)))
      # wrt Rad
      Rad_CHO = np.array(AUC_dict[sig_type][dim]['Rad']) - np.array(AUC_dict[sig_type][dim]['CHO'])
      Rad_FCO = np.array(AUC_dict[sig_type][dim]['Rad']) - np.array(AUC_dict[sig_type][dim]['FCO'])
      print('p value - Rad-CHO : {}'.format(np.mean(Rad_CHO<0)))
      print('p value - Rad-FCO : {}'.format(np.mean(Rad_FCO<0)))
      # wrt FCO
      FCO_CHO = np.array(AUC_dict[sig_type][dim]['FCO']) - np.array(AUC_dict[sig_type][dim]['CHO'])
      print('p value - FCO-CHO : {}'.format(np.mean(FCO_CHO<0)))

    print(out_table)