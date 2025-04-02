import os
import numpy as np

import scipy.io as sio
import heapq

import matplotlib.pyplot as plt
from scipy import stats, signal

import seaborn as sns
from matplotlib.patches import Rectangle

import pickle as pkl
from numpy import linalg as LA

save_root = '201_Visualization_maps/'


# Directory containing output files - use them for analysis
my_dir_root = save_root + '3D_Radiologist_data/'
print(os.listdir(my_dir_root))


screen_X = 2560 
screen_Y = 2048
screenSize=[2048, 2560]
pad_X = 0
pad_Y = 0
day='3D'
windowLevel=10987
windowWidth=11841

participants = []
for name in os.listdir('Dataset/Radiologists/'+day):
  if len(name.split('_'))>1:
    print(name)
    participants.append(name)
print('participants: {}'.format(participants))



root_dir = save_root + day + '_Radiologist_data/'
try:
  os.mkdir(root_dir)
except:
  pass
  
def dist_single(x, y):
  return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (5*(x[2]-y[2]))**2)

def get_min_dist(new_idx, idx_arr):
  min_dist = 1000000
  for row_idx in range(idx_arr.shape[0]):
    dist_ = dist_single(new_idx, idx_arr[row_idx])
    if dist_<min_dist:
      min_dist = dist_
  return min_dist

import time


def get_top_locations(q, cnt, flag_D):
  #print('Enter: get_top_locations, q.shape: {}'.format(q.shape))
  start_time = time.time()
  #q_flat = q.reshape(-1)

  # Downscale the input array
  step_x = 45 #11 #22 #45 #16 #45 #16
  step_y = 45 #11 #22 #45 #16 #45 #16
  if flag_D=='3D':
    step_z = 3 #4 #8 #4

  num_x = int(np.floor(q.shape[1]//step_x))
  num_y = int(np.floor(q.shape[0]//step_y))
  if flag_D=='3D':
    num_z = int(np.floor(q.shape[2]//step_z))
    max_arr = np.zeros((num_y,num_x,num_z))
    org_idx = np.zeros((num_y,num_x,num_z, 3))
  else:
    num_z = 1
    max_arr = np.zeros((num_y,num_x))
    org_idx = np.zeros((num_y,num_x, 2))


  for blk_y_ in range(num_y): #np.arange(0, q.shape[1], step_x):
    for blk_x_ in range(num_x): #np.arange(0, q.shape[0], step_y):
      for blk_z_ in range(num_z):
        blk_x = blk_x_*step_x
        blk_y = blk_y_*step_y
        if flag_D=='3D':
          blk_z = blk_z_*step_z
          max_val = np.max(q[blk_y:(blk_y+step_y), blk_x:(blk_x+step_x), blk_z:(blk_z+step_z)])
          max_idx = np.unravel_index(np.argmax(q[blk_y:(blk_y+step_y), blk_x:(blk_x+step_x), blk_z:(blk_z+step_z)]), (step_y, step_x, step_z))
          org_idx[blk_y_, blk_x_, blk_z_, 0] = int(max_idx[0] + blk_y)
          org_idx[blk_y_, blk_x_, blk_z_, 1] = int(max_idx[1] + blk_x)
          org_idx[blk_y_, blk_x_, blk_z_, 2] = int(max_idx[2] + blk_z)
          #print(org_idx[blk_y_, blk_x_, blk_z_])
          max_val_org = q[int(org_idx[blk_y_, blk_x_, blk_z_, 0]), int(org_idx[blk_y_, blk_x_, blk_z_, 1]), int(org_idx[blk_y_, blk_x_, blk_z_, 2])]
          max_arr[blk_y_, blk_x_, blk_z_] = max_val
        else:
          max_val = np.max(q[blk_y:(blk_y+step_y), blk_x:(blk_x+step_x)])
          max_idx = np.unravel_index(np.argmax(q[blk_y:(blk_y+step_y), blk_x:(blk_x+step_x)]), (step_y, step_x))
          org_idx[blk_y_, blk_x_, 0] = int(max_idx[0] + blk_y)
          org_idx[blk_y_, blk_x_, 1] = int(max_idx[1] + blk_x)
          max_val_org = q[int(org_idx[blk_y_, blk_x_, 0]), int(org_idx[blk_y_, blk_x_, 1])]
          max_arr[blk_y_, blk_x_] = max_val

  #print([blk_x, blk_y, blk_z, max_val, max_val_org, max_idx])
  #print(max_arr)


  # Find the max locations in max arr and transform coordinates back to original array
  q_flat = max_arr.reshape(-1)

  #a = numpy.array([1, 30, 12, 40, 15])
  max_values = heapq.nlargest(cnt, q_flat)
  max_indicies = heapq.nlargest(cnt, range(len(q_flat)), q_flat.take)
  #print(max_values)
  if flag_D=='3D':
    max_indicies_arr = np.zeros((cnt,3))
  else:
    max_indicies_arr = np.zeros((cnt,2))


  for idx_, idx in enumerate(max_indicies):
    new_idx = np.unravel_index(idx, max_arr.shape)
    #print(new_idx)
    # Find the right location in the original array
    if flag_D=='3D':
      newest_idx = org_idx[new_idx[0], new_idx[1], new_idx[2]]
    else:
      newest_idx = org_idx[new_idx[0], new_idx[1]]
    #print(newest_idx)
    #print(q[int(newest_idx[0]), int(newest_idx[1]), int(newest_idx[2])])
    max_indicies_arr[idx_, 0] = int(newest_idx[0])
    max_indicies_arr[idx_, 1] = int(newest_idx[1])
    if flag_D=='3D':
      max_indicies_arr[idx_, 2] = int(newest_idx[2])

  #print(max_indicies_arr)

  end_time = time.time()
  #print('Exit: get_top_locations ({})'.format(end_time-start_time))

  return max_indicies_arr.astype(np.uint32)

'''

# Plotting individual slices and the top-5 locations
#def Plot_Individual(stim_name, mat_contents, cx, cy, sl, f_name, title_txt, response, radiologist_name, phantom_dir, cho_path, fco_path, cnn_path, fixation_map, time_map, cho_map, fco_map, cnn_map):
def Plot_Individual(stim_name, mat_contents, cx, cy, sl, f_name, title_txt, response, radiologist_name, phantom_dir, fixation_map, time_map, cho_map, fco_map, cnn_map):

    stim_name_ = stim_name.split('\\')[-1]

    if True:
      for loc_idx in range(1):
        if response<3:
          c_ = 'red'
          decision_txt = 'Signal absent'
        else:
          c_ = 'green'
          decision_txt = 'Signal present'
        print(decision_txt)


        # Visualize the fixation and time density maps 
        for slice_idx in range(64): #np.arange(sl-8, sl+9): #range(64): #[sl]: #range(64):
          #print('slice idx: {}'.format(slice_idx))
          # Initialize the figure          
          fig = plt.figure(figsize=(24,8), dpi=100)
          q_r = 4
          q_c = 7
          #ax0 = plt.subplot2grid((q_r, q_c), (0, 1), rowspan=4)
          ax1 = plt.subplot2grid((q_r, q_c), (0, 0), rowspan=4)
          ax3a = plt.subplot2grid((q_r, q_c), (0, 1), rowspan=4)
          ax3b = plt.subplot2grid((q_r, q_c), (0, 2), rowspan=4) 
          ax4 = plt.subplot2grid((q_r, q_c), (0, 3), rowspan=4)
          ax5 = plt.subplot2grid((q_r, q_c), (0, 4), rowspan=4)  
          ax6 = plt.subplot2grid((q_r, q_c), (0, 5), rowspan=4)  
          
          # Plot the image with fixations and location of signal
          annot_kws={'fontsize':10, 'fontstyle':'italic','fontweight':'bold', 'color':'k', 'alpha':0.6, 'rotation':None,'verticalalignment':'center', 'backgroundcolor':'None'}
          cbar_kws={'label':'within 1 deg radius','orientation':'vertical', 'shrink':1,'extend':'max', 'ticks':[0, 0.5,1.0], 'extendfrac':0.1, 'drawedges':True }

          sns.heatmap(tiff[:,1000:,slice_idx], xticklabels=False, yticklabels=False, square=True, ax=ax1, cbar=False, cbar_kws=cbar_kws, annot_kws=annot_kws, cmap='gray')#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          ax1.set_title('slice: ' + str(slice_idx) + '\n'+'Phantom ('+f_name+ ', ' + 'slice: ' + str(sl) + ')\n'+'Response: '+str(response), fontsize=12, fontweight='bold')
          #print('signoi: {}'.format(signoi))
          if f_name!='NoLesion':
            print('Adding patch: f_name: {}, cx: {}, cy: {}'.format(f_name, cx, cy))
            ax1.add_patch(Rectangle((cx-50, cy-50), 100, 100, fill=False, color='r', lw=2))  


          'ax0:  Fixations made'
          rec = [] # record of useful fixation locations
          #sns.heatmap(tiff[:,1000:,sl], xticklabels=False, yticklabels=False, square=True, ax=ax0, cbar=False, cbar_kws=cbar_kws, annot_kws=annot_kws, vmin=0.0, vmax=1.0, cmap='gray') #cmap="YlGnBu"
          #ax0.set_title('Fixations', fontsize=12, fontweight='bold')


          annot_kws={'fontsize':4, 'fontstyle':'italic','fontweight':'bold', 'color':'k', 'alpha':0.6, 'rotation':None,'verticalalignment':'center', 'backgroundcolor':'None'}
          cbar_kws={'label':'within a radius','orientation':'vertical', 'shrink':0.75,'extend':'max', 'extendfrac':0.2, 'drawedges':True }

          # Plot the fixation density map
          ax3a.set_title('Fixation density', fontsize=12, fontweight='bold')
          sns.heatmap(fixation_map[:,:,slice_idx], cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax3a, cbar=True, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          if slice_idx==0:
            fixation_indicies_arr = get_top_locations(fixation_map, 5)
          for row_idx in range(fixation_indicies_arr.shape[0]):
            if fixation_indicies_arr[row_idx,2]==slice_idx:
              #print('slice: {}, Adding in fixation map'.format(slice_idx))
              ax3a.add_patch(Rectangle((fixation_indicies_arr[row_idx,1]-50, fixation_indicies_arr[row_idx,0]-50), 100, 100, fill=False, color='r', lw=2))  

          # Plot the density map of time-spent
          ax3b.set_title('Time-Spent density', fontsize=12, fontweight='bold')
          sns.heatmap(time_map[:,:,slice_idx], cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax3b, cbar=True, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          if slice_idx==0:
            time_indicies_arr = get_top_locations(time_map, 5)
          for row_idx in range(time_indicies_arr.shape[0]):
            if time_indicies_arr[row_idx,2]==slice_idx:
              #print('slice: {}, Adding in time map'.format(slice_idx))
              ax3b.add_patch(Rectangle((time_indicies_arr[row_idx,1]-50, time_indicies_arr[row_idx,0]-50), 100, 100, fill=False, color='r', lw=2))  

          # Title - name of phantom, name of radiologist, response, slice with signal



          # CHO - Just plot the response
          sns.heatmap(cho_map[:,:,slice_idx], cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax4, cbar=True, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          ax4.set_title('CHO', fontsize=12, fontweight='bold')
          if slice_idx==0:
            cho_indicies_arr = get_top_locations(cho_map, 5)
          for row_idx in range(cho_indicies_arr.shape[0]):
            if cho_indicies_arr[row_idx,2]==slice_idx:
              #print('slice: {}, Adding in cho map'.format(slice_idx))
              ax4.add_patch(Rectangle((cho_indicies_arr[row_idx,1]-50, cho_indicies_arr[row_idx,0]-50), 100, 100, fill=False, color='r', lw=2))  


          # FCO - Just plot the response
          sns.heatmap(fco_map[:,:,slice_idx], cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax5, cbar=True, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          ax5.set_title('FCO', fontsize=12, fontweight='bold')
          if slice_idx==0:
            fco_indicies_arr = get_top_locations(fco_map, 5)
          for row_idx in range(fco_indicies_arr.shape[0]):
            if fco_indicies_arr[row_idx,2]==slice_idx:
              #print('slice: {}, Adding in fco map'.format(slice_idx))
              ax5.add_patch(Rectangle((fco_indicies_arr[row_idx,1]-50, fco_indicies_arr[row_idx,0]-50), 100, 100, fill=False, color='r', lw=2))  


          # CNN - Just plot the softmax
          sns.heatmap(cnn_map[:,:,slice_idx], cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax6, cbar=True, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          ax6.set_title('CNN', fontsize=12, fontweight='bold')
          if slice_idx==0:
            cnn_indicies_arr = get_top_locations(cnn_map, 5)
          for row_idx in range(cnn_indicies_arr.shape[0]):
            if cnn_indicies_arr[row_idx,2]==slice_idx:
              #print('slice: {}, Adding in cnn map'.format(slice_idx))
              ax6.add_patch(Rectangle((cnn_indicies_arr[row_idx,1]-50, cnn_indicies_arr[row_idx,0]-50), 100, 100, fill=False, color='r', lw=2))  


          # Save and close the figure                  
          plt.tight_layout()
          plt.savefig(phantom_dir+radiologist_name+'_slice_'+str(slice_idx)+'.png')

          plt.close()

    
    return

'''





save_dir = save_root+'September_2023/'
try:
  os.mkdir(save_dir) 
except:
  pass
perc_values = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45]#, 50]
# First key - phantom_name, Second key - radiologist, Third key - Correlation, Response and Fixation/Time map?
Analysis_4_dict = {}
Analysis_4x_dict = {}


def phantom_data(stim_name, f_name):
    stim_name_ = stim_name.split('\\')

    # Read the corresponding phantom    
    orig_mat_ = sio.loadmat('Dataset/RSNA2017/AJ_Combined/' + f_name + '/' + stim_name_[-1] + '/phantom.mat')
    tiff = orig_mat_['tiff']    #
    #tiff = tiff/tiff.max()
    if stim_name_[-2]!='NoLesion':
      file1 = open('Dataset/RSNA2017/AJ_Combined/' + f_name + '/' + stim_name_[-1] + '/locations.txt',"r+") 
      z = file1.read()
      cx = 10*float(z.split()[0])-1000
      cy = 10*float(z.split()[2])
      sl = int(float(z.split()[1]))
      print('cx: {}, cy: {}, slice: {}'.format(cx, cy, sl))
      title_txt = 'Signal present'
    else:
      sl=31
      cx = -1000
      cy = -1000
      title_txt = 'Signal absent'

    return tiff, cx, cy, sl, title_txt



def compute_norm(inp_map):
  #return LA.norm(inp_map)
  return np.sum(inp_map)





if True:
  phantom_names = []
  phantom_dict = {}
  for P_idx in range(len(participants)):
    # Image index seen by this radiologist
    for I_idx in np.arange(1,29): #(14,15): #(1,29): #(10,11): #(11,12):#29):#13,14):#1,29): #4): #range(11,12): #28
      print('\nImage_idx: {}, Participant_idx: {}'.format(I_idx, P_idx))
      # Load the meta data for Radiologist and this phantom
      mat_contents = sio.loadmat('Dataset/Radiologists/'+day+'/'+participants[P_idx]+'/data'+str(I_idx)+'.mat')
      # Response by the radiologist
      response = mat_contents['resp'][0][0]['RATING'][0][0]
      # Name of the phantom shown
      stim_name = mat_contents['stimInfo'][0][0]['imgName'][0] #mat_contents['stimInfo'][0][0][8][0]
      # Corresponding location of cho and npwe responses
      stim_name_ = stim_name.split('\\')
      if stim_name_[-2]=='NoLesion':
        f_name=stim_name_[-2]
      elif stim_name_[-2]=='calc_041':
        f_name=stim_name_[-2][:-4]
      elif stim_name_[-2]=='masses_143':
        f_name=stim_name_[-2][:-6]

      #f_name, cho_path, npwe_path, cho_th, fco_path, cnn_path = phantom_name(stim_name)
      print('stimulus_name: {}, signal_type: {},  Response: {}'.format(stim_name, f_name, response))
      #print(cho_path, fco_path, cnn_path)

      ph_name = stim_name.split('\\')[-1]
      if f_name!='calc' and f_name!='mass':
        phantom_names.append(ph_name)
        phantom_dict[ph_name] = stim_name
        print('\n\nphantom_names ({}): {}'.format(len(set(phantom_names)), set(phantom_names)))
      continue

  phantom_names = set(phantom_names)

#cue = mat_contents['stimInfo']['hfSignal'][0,0][0][0]
#if (cue==0 and sig_type=='mass') or (cue==1 and sig_type=='calc'):
#  keep_img = True


for cond_idx in [2,3]: #range(4):
  if cond_idx==0:
    model_type = 'max'
  elif cond_idx==1:
    model_type = 'sum'
  elif cond_idx==2:
    model_type = 'calc'
  elif cond_idx==3:
    model_type = 'mass'


  Analysis_2_dict = {}
  for flag_D in ['3D']: #['2D', '3D']:
    Analysis_2_dict[flag_D] = {}
    for percn in perc_values: #[1,10,20,30,40,50,60,70,80,90,100]: #np.arange(1,101,1): #[1, 5, 25]:
      Analysis_2_dict[flag_D][str(percn)] = {}
      for sig_type in ['NoLesion']:#, 'calc', 'mass']:
        Analysis_2_dict[flag_D][str(percn)][sig_type] = {}
      for MO_name in ['cho', 'fco', 'cnn']:
        for sig_type in ['NoLesion']:#, 'calc', 'mass']:
          Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name] = {}
          #Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time'] = []
          Analysis_2_dict[flag_D][str(percn)][sig_type][MO_name]['time_dict'] = {}


  Analysis_3_dict = {}
  for percn in ['3D']: #['2D', '3D']:
    Analysis_3_dict[str(percn)] = {}
    for sig_type in ['NoLesion']:#, 'calc', 'mass']:
      Analysis_3_dict[str(percn)][sig_type] = {}
    for MO_name in ['cho', 'fco', 'cnn']:
      for sig_type in ['NoLesion']:#, 'calc', 'mass']:
        Analysis_3_dict[str(percn)][sig_type][MO_name] = {}
        #Analysis_3_dict[str(percn)][sig_type][MO_name]['time'] = []
        Analysis_3_dict[str(percn)][sig_type][MO_name]['time_dict'] = {}
        Analysis_3_dict[str(percn)][sig_type][MO_name]['time_time'] = {}
        #Analysis_3_dict[str(percn)][sig_type][MO_name]['time_individual'] = []


  cnt = 0
  #for P_idx in range(len(participants)):
  phantom_names_sorted = sorted(list(phantom_names))
  for idx, name in enumerate(phantom_names_sorted):
    print([idx, name])
  #for ph_name in list(phantom_names): #[:1]:
  for ph_idx, ph_name in enumerate(phantom_names_sorted):
    #if ph_idx!=4:
    #  continue
    cnt+=1
    flag_exit = 0
    sc_global, lc_tm_global, lc_slice_global = [], [], []
    Analysis_4_dict[ph_name] = {}
    valid_participants = []
    for P_idx in range(len(participants)):
      # Image index seen by this radiologist
      for I_idx in np.arange(1,29): #(14,15): #(1,29): #(10,11): #(11,12):#29):#13,14):#1,29): #4): #range(11,12): #28
        mat_contents = sio.loadmat('Dataset/Radiologists/'+day+'/'+participants[P_idx]+'/data'+str(I_idx)+'.mat')
        response = mat_contents['resp'][0][0]['RATING'][0][0]
        stim_name = mat_contents['stimInfo'][0][0]['imgName'][0] #mat_contents['stimInfo'][0][0][8][0]
        stim_name_ = stim_name.split('\\')
        cue = mat_contents['stimInfo']['hfSignal'][0,0][0][0]
        if stim_name_[-2]=='NoLesion':
          f_name=stim_name_[-2]
        elif stim_name_[-2]=='calc_041':
          f_name=stim_name_[-2][:-4]
        elif stim_name_[-2]=='masses_143':
          f_name=stim_name_[-2][:-6]

        ph_name_ = stim_name.split('\\')[-1]
        if f_name!='NoLesion' or ph_name!=ph_name_:
          continue

        #*********** NEW *************
        if (cue==1 and model_type=='mass') or (cue==0 and model_type=='calc'):
          continue

        #print('######### {} - {} - {}'.format(participants[P_idx], I_idx, ph_name_))
        valid_participants.append(participants[P_idx])
        #if f_name=='NoLesion' and ph_name==ph_name_:
        #  flag_exit = 1
        #  break
      #if flag_exit==1:
      #  break

    print('cnt: {}, ph_name: {}, valid_participants: {}'.format(cnt, ph_name, valid_participants))

    # Skip if no radiologist saw this phantom in this condition
    if len(valid_participants)==0:
      continue

    print('Will be processed!')
    #continue

    # Extract phantom data
    tiff, cx, cy, sl, title_txt = phantom_data(phantom_dict[ph_name], 'NoLesion')
    #sc_global = np.concatenate(sc_global, axis=0)

    # Mask generation
    tiff_bool = tiff[:,1000:]/16384.0
    tiff_kernel = (1/400)*np.ones((20,20,1))
    tiff_bool = signal.fftconvolve(tiff_bool, tiff_kernel, mode='same')
    tiff_bool = tiff_bool>0.4272
    tiff_bool[:,-80:]=0
    tiff_area = np.sum(tiff_bool)/(2048*792*64)
    print('****** AJ: Name: {}, tiff.shape: {}, percentage_area: {}'.format(ph_name, tiff_bool.shape, tiff_area))



    t_start = time.time()
    phantom_name = ph_name #stim_name_[-1]
    sig_type = 'NoLesion' #f_name
    f_name = sig_type
    phantom_dir = my_dir_root+sig_type+'/'+phantom_name+'/'
    radiologist_name = 'combined' #participants[P_idx]
    print('Loading phantom from {}'.format(phantom_dir))
    if True: #try:
      #fixation_map = np.load(phantom_dir+radiologist_name+'_'+'fixation_map.npy')
      #time_map = np.load(phantom_dir+radiologist_name+'_'+'time_map.npy')
      cho_map = np.load(phantom_dir+model_type+'_'+'cho_map.npy')
      fco_map = np.load(phantom_dir+model_type+'_'+'fco_map.npy')
      cnn_map = np.load(phantom_dir+model_type+'_'+'cnn_map.npy')
      # Read individual fixation and time maps
      #participants_list = []
      time_map_dict = {}
      for radiologist_name in valid_participants:
        try:
          t_map = np.load(phantom_dir+radiologist_name+'_'+'time_map.npy')
          t_map *= tiff_bool
          print('np.sum(t_map): {}, np.sum(np.isnan(t_map)): {}'.format(np.sum(t_map), np.sum(np.isnan(t_map))))
          if np.sum(t_map)>0:
            t_map = (t_map/np.sum(t_map))*100.0
            #valid_time_maps.append(t_map)
            #participants_list.append(radiologist_name)
            time_map_dict[radiologist_name] = t_map
        except:
          print('FIX ERROR !!')
          pass
      #cnn_map = cnn_map[:,:-1]
      #cnn_map[:,-80:,:] = 0
      #fixation_map *= tiff_bool
      #time_map *= tiff_bool
      cho_map *= tiff_bool
      fco_map *= tiff_bool
      cnn_map *= tiff_bool
      # Normalize
      #fixation_map = (fixation_map/np.sum(fixation_map))*100.0
      #time_map = (time_map/np.sum(time_map))*100.0
      cho_map = cho_map/np.sum(cho_map)
      fco_map = fco_map/np.sum(fco_map)
      cnn_map = cnn_map/np.sum(cnn_map)
      #print([cho_map.shape, fco_map.shape, cnn_map.shape, len(valid_fix_maps), len(valid_time_maps)])
    #except:
    #  print('Not found!!')
    #  break
    t_end = time.time()

    radiologist_name = 'combined' #participants[P_idx]
    print('\ncnt: {}, DONE with LOADING maps, time taken: {}'.format(cnt, t_end-t_start))

    


    # Analysis-2 : For a given percentage, get locations from CHO / FCO / CNN maps and see how many fixations are made in those regions or the time spent
    print('\n Analysis-2')
    names_list = []
    for radiologist_name in time_map_dict.keys():
      names_list.append(radiologist_name)
    for flag_D in range(1): #['2D', '3D']:
      for percn in perc_values: #[1,10,20,30,40,50,60,70,80,90,100]: #np.arange(1,101,1): #[1, 5, 25]:
        for MO_name in ['cho', 'fco', 'cnn']:
          if MO_name=='cho':
            mo_map = cho_map
          elif MO_name=='fco':
            mo_map = fco_map
          elif MO_name=='cnn':
            mo_map = cnn_map
          # 3D analysis
          mo_median = np.percentile(mo_map, 100-percn)
          mo_true = mo_map>mo_median
          a, b, c = mo_map.shape
          mo_percentage = np.sum(mo_true)/(a*b*c)
          # Compute fixations and time spent in those locations
          #time_spent = np.sum(time_map[mo_true])/np.sum(time_map)
          #Analysis_2_dict['3D'][str(percn)][f_name][MO_name]['time'].append(time_spent)

          # Dictionary - radiologist - phantom name - percentage time spent at top locations of this MO
          for rad_idx, radiologist_name in enumerate(names_list):
            t_map = time_map_dict[radiologist_name] #valid_time_maps[rad_idx]
            # Compute percentage of time spent
            time_spent = np.sum(t_map[mo_true])/np.sum(t_map) 
            # Add the entry in the dictionary
            if not radiologist_name in Analysis_2_dict['3D'][str(percn)][f_name][MO_name]['time_dict'].keys():
              Analysis_2_dict['3D'][str(percn)][f_name][MO_name]['time_dict'][radiologist_name] = {}
            Analysis_2_dict['3D'][str(percn)][f_name][MO_name]['time_dict'][radiologist_name][ph_name] = time_spent

          print('3D: {}: median: {:.2e}, percentage: {:.2e}, #time: {:.2e}'.format(MO_name, mo_median, mo_percentage, time_spent))
    #break
    with open(save_dir+model_type+'_type_2_3D_analysis.pkl', 'wb') as f:
      pkl.dump(Analysis_2_dict, f)













    # Analysis-3 : Dot product between the fixation / time-spent and the model observer
    print('\n Analysis-3')
    #time_map_nonzero = time_map[tiff_bool]
    valid_time_maps_nonzero = []
    #for t_map in valid_time_maps:
    #  valid_time_maps_nonzero.append(t_map[tiff_bool])
    names_list = []
    for radiologist_name in time_map_dict.keys():
      t_map = time_map_dict[radiologist_name]
      valid_time_maps_nonzero.append(t_map[tiff_bool])
      names_list.append(radiologist_name)



    #valid_time_maps = []
    #print('****** time_map_nonzero.shape: {}'.format(time_map_nonzero.shape))
    for percn in ['3D']: #['2D', '3D']:
      if percn=='3D':
        #time_map_norm = time_map_nonzero #.reshape(-1) #/compute_norm(time_map) #LA.norm(time_map)
        #t_corrs = []
        # Create the combined map using the rest of the maps
        for t_idx, t_map in enumerate(valid_time_maps_nonzero):
          radiologist_name = names_list[t_idx]
          c_map = 0
          for t_idx_, t_map_ in enumerate(valid_time_maps_nonzero):
            if t_idx==t_idx_:
              continue
            c_map+=t_map_
          if len(valid_time_maps_nonzero)>1:
            c_map = c_map/(len(valid_time_maps_nonzero)-1)
            '''
            # Debug info
            print('Correlation#1: t_map.shape: {} ({} - {}), c_map.shape: {} ({} - {})'.format(t_map.shape, np.min(t_map), np.max(t_map), c_map.shape, np.min(c_map), np.max(c_map)))
            debug_corr = np.corrcoef(t_map, c_map)
            print(debug_corr.shape)
            print(debug_corr)
            debug_corr = np.corrcoef(t_map, c_map/np.sum(c_map))
            print(debug_corr)
            '''
            t_corr = np.corrcoef(t_map, c_map)[1,0]
            #t_corrs.append(t_corr)
            if radiologist_name not in Analysis_3_dict[str(percn)][f_name][MO_name]['time_time'].keys():
              Analysis_3_dict[str(percn)][f_name][MO_name]['time_time'][radiologist_name] = {}
            Analysis_3_dict[str(percn)][f_name][MO_name]['time_time'][radiologist_name][ph_name] = t_corr
            print('{}: len(valid_time_maps_nonzero): {}, t_corr: {}'.format(t_idx, len(valid_time_maps_nonzero), t_corr))
        #Analysis_3_dict[str(percn)][f_name][MO_name]['time_time'].append(t_corrs)

        

      for MO_name in ['cho', 'fco', 'cnn']:
        if MO_name=='cho':
          mo_map = cho_map #[tiff_bool]
        elif MO_name=='fco':
          mo_map = fco_map #[tiff_bool]
        elif MO_name=='cnn':
          mo_map = cnn_map #[tiff_bool]
        if percn=='3D':
          mo_map_norm = mo_map[tiff_bool] #.reshape(-1) #/compute_norm(mo_map) #LA.norm(mo_map)
          #time_corrcoef = np.corrcoef(time_map_norm, mo_map_norm)[1,0]
          #Analysis_3_dict[str(percn)][f_name][MO_name]['time'].append(time_corrcoef) #np.sum(time_map_norm*mo_map_norm))
          t_corrs = []
          for rad_idx, t_map in enumerate(valid_time_maps_nonzero):
            #print('Correlation#1: t_map.shape: {} ({} - {}), mo_map_norm.shape: {} ({} - {})'.format(t_map.shape, np.min(t_map), np.max(t_map), mo_map_norm.shape, np.min(mo_map_norm), np.max(mo_map_norm)))
            t_corr = np.corrcoef(t_map, mo_map_norm)[1,0]
            t_corrs.append(t_corr)
            # Get the radiologist name and check if entry is present 
            #radiologist_name = participants_list[rad_idx]
            radiologist_name = names_list[rad_idx]
            if not radiologist_name in Analysis_3_dict[str(percn)][f_name][MO_name]['time_dict'].keys():
              Analysis_3_dict[str(percn)][f_name][MO_name]['time_dict'][radiologist_name] = {}
            Analysis_3_dict[str(percn)][f_name][MO_name]['time_dict'][radiologist_name][ph_name] = t_corr
          #Analysis_3_dict[str(percn)][f_name][MO_name]['time_individual'].append(t_corrs)

        #print('{}: time norm: {:.2e}'.format(percn, np.mean(Analysis_3_dict[str(percn)][f_name][MO_name]['time'])))
    #break
    with open(save_dir+model_type+'_type_3_3D_analysis.pkl', 'wb') as f:
      pkl.dump(Analysis_3_dict, f)


    #break
 

