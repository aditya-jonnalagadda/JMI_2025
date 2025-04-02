import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from numpy import linalg as LA
import scipy.io as sio

from matplotlib import rc
rc('text', usetex=False)

import os
from scipy import ndimage, signal

from scipy.ndimage import gaussian_filter

pixels_per_degree = 45



def phantom_name(stim_name):
    
    stim_name_ = stim_name.split('\\')
    cn_dir = 'Files_sorted/' 
    cn_dir_ = cn_dir
    if stim_name_[-2]=='NoLesion':
      f_name=stim_name_[-2]
      cho_path = [cn_dir+'CHO/calc/', cn_dir+'CHO/mass/']
      fco_path = [cn_dir+'FCO/calc/', cn_dir+'FCO/mass/']
      cnn_path = [cn_dir+'CNN/calc/', cn_dir+'CNN/mass/']
      npwe_path = [cn_dir_+'NPWE/calc/', cn_dir_+'NPWE/mass/']
      cho_th = [0.2889, 540.4288]
    elif stim_name_[-2]=='calc_041':
      f_name=stim_name_[-2][:-4]
      cho_path = [cn_dir+'CHO/3D/CALC/Signal/']
      fco_path = [cn_dir+'FCO/3D/CALC/Signal/']
      cnn_path = [cn_dir+'CNN/3D/CALC/Signal/']
      npwe_path = [cn_dir_+'NPWE/2D/CALC/Signal/']
      cho_th = [0.2889]
    elif stim_name_[-2]=='masses_143':
      f_name=stim_name_[-2][:-6]
      cho_path = [cn_dir+'CHO/3D/MASS/Signal/']
      fco_path = [cn_dir+'FCO/3D/MASS/Signal/']
      cnn_path = [cn_dir+'CNN/3D/MASS/Signal/']
      npwe_path = [cn_dir_+'NPWE/2D/MASS/Signal/']
      cho_th = [540.4288]

    return f_name, cho_path, npwe_path, cho_th, fco_path, cnn_path

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

def phantom_saccades(mat_contents):
    sacStart=mat_contents['sacInfo'][0][0]['startTime']
    sacEnd=mat_contents['sacInfo'][0][0]['endTime']
    sacX = mat_contents['sacInfo'][0][0]['xEnd'][0]
    sacY = mat_contents['sacInfo'][0][0]['yEnd'][0]
    drill = mat_contents['stimInfo'][0][0]['drillingMovements']
    lesionLocation = mat_contents['stimInfo'][0][0]['locations']

    sacEnd = sacEnd[sacEnd!=0]*86400

    return sacStart, sacEnd, sacX, sacY, drill, lesionLocation


def process_drills(drill, sacEnd):
    minDistance=np.Inf
    changeTime = []
    try:
      if day=='3D':
        drill[:,0]=drill[:,0]*86400

        direction=-1
        changeTime=[]
        changes=0
        for d in range(drill.shape[0]):
          if drill[d,1]!=direction:
            direction=direction*-1
            changes=changes+1
            changeTime.append(drill[d,0])
    except:
      #continue
      pass
      

    allTimes=[]
    allTimes.extend(list(sacEnd))
    if day=='3D':
      allTimes.extend(list(drill[:,0]))
    allTimes = np.sort(allTimes)
    print('Done with process_drills')

    return changeTime, allTimes


def process_eyemovements(sacEnd, sacX, sacY, drill, lesionLocation, ratioImgScreen, allTimes):

    timePrevSlice=allTimes[0]
    currentSlice=0

    sc = []
    lc_tm = []
    lc_slice = []
    for ct in range(1,len(allTimes)):
      curr_idx = len(sacEnd[sacEnd<allTimes[ct]])-1

      #print('process_eyemovements:  ct: {}, curr_idx: {}'.format(ct, curr_idx))

      # For the 3D case
      if day=='3D':
        cd=drill[:,0]<=allTimes[ct]
      nextFixation=[sacX[curr_idx], sacY[curr_idx]]
      if LA.norm(np.array([nextFixation[0]-ratioImgScreen[0]*lesionLocation[0][0]*10, nextFixation[1]-ratioImgScreen[1]*lesionLocation[0][2]*10]))<200:
        #print('Close to the signal!!!!')
        nextFixation=[ratioImgScreen[0]*lesionLocation[0][0]*10, ratioImgScreen[1]*lesionLocation[0][2]*10] #ratioImgScreen.*lesionLocation
      sc.append(nextFixation)
      lc_tm.append(allTimes[ct]-allTimes[ct-1])
      if day=='3D':
        lc_slice.append(sum(drill[cd,1]))
      else:
        lc_slice.append(sl)

      #print('curr_idx: {}, ct: {}, nextFixation: {}'.format(curr_idx, ct, nextFixation))

    #print('AJ: lc_slice: {}'.format(lc_slice))
    sc = np.stack(sc, axis=0)
    #print('process eye-movements: sc: {}'.format(sc))

    return sc, lc_tm, lc_slice



'''
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQqq
QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ

'''

def get_max_location(tiff_ds):
  z_flat = tiff_ds.reshape(-1)
  max_indicies = np.argpartition(z_flat, -5)[-5:][::-1]
  print('max_indicies: {}'.format(max_indicies))
  max_indicies_arr = np.zeros((5,3))
  for row_idx, idx in enumerate(max_indicies):
    max_indicies_arr[row_idx] = np.unravel_index(idx, tiff_ds.shape)

  return max_indicies_arr


def Generate_human_maps_SMALL(stim_name, sc, lc_tm, lc_slice, ratioImgScreen, mat_contents, cx, cy, sl, f_name, title_txt, response, radiologist_name, phantom_dir, tiff_bool):

    stim_name_ = stim_name.split('\\')[-1]
    print('Generate_human_maps: stim_name_: {}, slice: {}'.format(stim_name_, sl))
    plotSC = np.zeros(sc.shape)
    plotSC[:,0] = sc[:,0]/ratioImgScreen[0]
    plotSC[:,1] = sc[:,1]/ratioImgScreen[1]


    for loc_idx in range(1):#len(cho_th)):
        if response<3:
          c_ = 'red'
          decision_txt = 'Signal absent'
        else:
          c_ = 'green'
          decision_txt = 'Signal present'
        print(decision_txt)


        rec = [] # record of useful fixation locations
        patch_cnt=0
        #print('**** plotSC: {}'.format(plotSC))
        print('**** len(plotSC): {}, max(plotSC[0]): {}, max(plotSC[1]): {}, tiff.shape: {}'.format(len(plotSC), max(plotSC[:,0]), max(plotSC[:,1]), tiff.shape))
        plt_cnt = 0
        for idx_ in range(1,len(plotSC)):
          if plotSC[idx_,0]>1000 and plotSC[idx_,0]<1792 and plotSC[idx_,1]>0 and plotSC[idx_,1]<2048:
            plt_cnt += 1

        print('Number of points plotted: {}'.format(plt_cnt))



        for density in [1]: #range(2):
          if density==0:
            title = 'Fixation density'
          else:
            title = 'Density of time spent'
          print('\n' + title)

          # PLOT#3
          rec = [] # record of useful fixation locations
          tiff_zeros = np.zeros((2048,792, 64))
          patch_cnt=0
          print('**** len(plotSC): {}, max(plotSC[0]): {}, max(plotSC[1]): {}, tiff.shape: {}'.format(len(plotSC), max(plotSC[:,0]), max(plotSC[:,1]), tiff.shape))
          print('**** len(lc_tm): {}'.format(len(lc_tm)))
          
          plt_cnt = 0
          for idx_ in range(1,len(plotSC)):
            y_idx = int(plotSC[idx_,1])
            x_idx = int(plotSC[idx_,0]-1000)
            if plotSC[idx_,0]>=1000 and plotSC[idx_,0]<1792 and plotSC[idx_,1]>=0 and plotSC[idx_,1]<2048:
              plt_cnt += 1
              if density==0:
                # Density - fixations
                tiff_zeros[int(plotSC[idx_,1]), int(plotSC[idx_,0]-1000), int(lc_slice[idx_])] += 1
              else:
                # Density - time taken
                tiff_zeros[int(plotSC[idx_,1]), int(plotSC[idx_,0]-1000), int(lc_slice[idx_])] += lc_tm[idx_]


          #sns.heatmap(tiff_zeros, cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax3, cbar=True, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          print('Number of points plotted: {}'.format(plt_cnt))
          print('tiff_zeros: sum: {}, max: {}'.format(np.sum(tiff_zeros), np.max(tiff_zeros)))

          # Create the density map
          tiff_ds = np.zeros((2048, 792, 64))
          w_2 = 45 #11 #22 #45 #11 #22 #7 #23 #45 #23
          #w_2_ = -22 #-7 #-23 #-45 #-23
          z_2 = 3 #2 #4 #2 #2


          print('before convolution: input.shape: {}'.format(tiff_zeros.shape))
          #tiff_ds = ndimage.convolve(tiff_zeros, dist_bool, mode='constant', cval=0.0)
          #tiff_ds = signal.fftconvolve(tiff_zeros, dist_bool, mode='same')
          tiff_zeros *= tiff_bool
          tiff_ds = gaussian_filter(tiff_zeros, sigma=[w_2, w_2, z_2])
          print('******Convolution output: input.shape: {}, output.shape: {}'.format(tiff_zeros.shape, tiff_ds.shape))
          #continue

          # Find top-K max locations
          #cnt_K = 50
          z_flat = tiff_ds.reshape(-1)
          max_indicies = np.argpartition(z_flat, -5)[-5:][::-1]
          print('max_indicies: {}'.format(max_indicies))
          max_indicies_arr = np.zeros((5,3))
          for row_idx, idx in enumerate(max_indicies):
            max_indicies_arr[row_idx] = np.unravel_index(idx, tiff_ds.shape)

          if density==0:
            fixation_map = tiff_ds
            fixation_indicies_arr = max_indicies_arr
            print('AJ: Assigning to fixation map, indicies: {}'.format(fixation_indicies_arr))
          else:
            time_map = tiff_ds
            time_indicies_arr = max_indicies_arr
            print('AJ: Assigning to time map, indicies: {}'.format(time_indicies_arr))

          # Save the mat files
          print('AJ: storing at location: {}'.format(phantom_dir+radiologist_name))
          if density==0:
            with open(phantom_dir+radiologist_name+'_fixation_map.npy', 'wb') as f:
              np.save(f, fixation_map)
          else:
            with open(phantom_dir+radiologist_name+'_time_map.npy', 'wb') as f:
              np.save(f, time_map)

          print('saved file at directory: {}'.format(phantom_dir+radiologist_name))

    return #fixation_map, time_map, cho_max, fco_max, cnn_max


def Generate_human_maps(stim_name, sc, lc_tm, lc_slice, ratioImgScreen, mat_contents, cx, cy, sl, f_name, title_txt, response, radiologist_name, phantom_dir, cho_path, fco_path, cnn_path, tiff_bool):

    stim_name_ = stim_name.split('\\')[-1]
    print('Generate_human_maps: stim_name_: {}, slice: {}'.format(stim_name_, sl))
    plotSC = np.zeros(sc.shape)
    plotSC[:,0] = sc[:,0]/ratioImgScreen[0]
    plotSC[:,1] = sc[:,1]/ratioImgScreen[1]


    for loc_idx in range(1):#len(cho_th)):
        if response<3:
          c_ = 'red'
          decision_txt = 'Signal absent'
        else:
          c_ = 'green'
          decision_txt = 'Signal present'
        print(decision_txt)


        rec = [] # record of useful fixation locations
        patch_cnt=0
        #print('**** plotSC: {}'.format(plotSC))
        print('**** len(plotSC): {}, max(plotSC[0]): {}, max(plotSC[1]): {}, tiff.shape: {}'.format(len(plotSC), max(plotSC[:,0]), max(plotSC[:,1]), tiff.shape))
        plt_cnt = 0
        for idx_ in range(1,len(plotSC)):
          if plotSC[idx_,0]>1000 and plotSC[idx_,0]<1792 and plotSC[idx_,1]>0 and plotSC[idx_,1]<2048:
            plt_cnt += 1

        print('Number of points plotted: {}'.format(plt_cnt))



        for density in [1]:#range(2):
          if density==0:
            title = 'Fixation density'
          else:
            title = 'Density of time spent'
          print('\n' + title)

          # PLOT#3
          rec = [] # record of useful fixation locations
          tiff_zeros = np.zeros((2048,792, 64))
          patch_cnt=0
          print('**** len(plotSC): {}, max(plotSC[0]): {}, max(plotSC[1]): {}, tiff.shape: {}'.format(len(plotSC), max(plotSC[:,0]), max(plotSC[:,1]), tiff.shape))
          print('**** len(lc_tm): {}'.format(len(lc_tm)))
          
          plt_cnt = 0
          for idx_ in range(1,len(plotSC)):
            y_idx = int(plotSC[idx_,1])
            x_idx = int(plotSC[idx_,0]-1000)
            if plotSC[idx_,0]>=1000 and plotSC[idx_,0]<1792 and plotSC[idx_,1]>=0 and plotSC[idx_,1]<2048:
              plt_cnt += 1
              if density==0:
                # Density - fixations
                tiff_zeros[int(plotSC[idx_,1]), int(plotSC[idx_,0]-1000), int(lc_slice[idx_])] += 1
              else:
                # Density - time taken
                tiff_zeros[int(plotSC[idx_,1]), int(plotSC[idx_,0]-1000), int(lc_slice[idx_])] += lc_tm[idx_]


          #sns.heatmap(tiff_zeros, cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax3, cbar=True, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          print('Number of points plotted: {}'.format(plt_cnt))
          print('tiff_zeros: sum: {}, max: {}'.format(np.sum(tiff_zeros), np.max(tiff_zeros)))

          # Create the density map
          tiff_ds = np.zeros((2048, 792, 64))
          w_2 = 45 #11 #22 #45 #11 #22 #7 #23 #45 #23
          #w_2_ = -22 #-7 #-23 #-45 #-23
          z_2 = 3 #2 #4 #2 #2

          '''
          # Compute the distances apriori - tells which pixels fall inside the circle
          dist_map = {} #np.zeros((5,5))
          dist_bool = np.zeros((2*w_2+1,2*w_2+1,2*z_2+1))
          for c_z in np.arange(0,2*z_2+1):
            for c_y in np.arange(w_2_,w_2+1):
              dist_map[c_y] = {}
              for c_x in np.arange(w_2_,w_2+1):
                dist_map[c_y][c_x] = np.sqrt(c_y**2 + c_x**2)
                if dist_map[c_y][c_x]<=w_2:
                  dist_bool[w_2+c_y, w_2+c_x, c_z] = 1
          '''

          print('before convolution: input.shape: {}'.format(tiff_zeros.shape))
          #tiff_ds = ndimage.convolve(tiff_zeros, dist_bool, mode='constant', cval=0.0)
          #tiff_ds = signal.fftconvolve(tiff_zeros, dist_bool, mode='same')
          tiff_zeros *= tiff_bool
          tiff_ds = gaussian_filter(tiff_zeros, sigma=[w_2, w_2, z_2])
          print('******Convolution output: input.shape: {}, output.shape: {}'.format(tiff_zeros.shape, tiff_ds.shape))
          #continue

          # Find top-K max locations
          #cnt_K = 50
          z_flat = tiff_ds.reshape(-1)
          max_indicies = np.argpartition(z_flat, -5)[-5:][::-1]
          print('max_indicies: {}'.format(max_indicies))
          max_indicies_arr = np.zeros((5,3))
          for row_idx, idx in enumerate(max_indicies):
            max_indicies_arr[row_idx] = np.unravel_index(idx, tiff_ds.shape)

          if density==0:
            fixation_map = tiff_ds
            fixation_indicies_arr = max_indicies_arr
            print('AJ: Assigning to fixation map, indicies: {}'.format(fixation_indicies_arr))
          else:
            time_map = tiff_ds
            time_indicies_arr = max_indicies_arr
            print('AJ: Assigning to time map, indicies: {}'.format(time_indicies_arr))

          # Save the mat files
          print('AJ: storing at location: {}'.format(phantom_dir+radiologist_name))
          if density==0:
            with open(phantom_dir+radiologist_name+'_fixation_map.npy', 'wb') as f:
              np.save(f, fixation_map)
          else:
            with open(phantom_dir+radiologist_name+'_time_map.npy', 'wb') as f:
              np.save(f, time_map)


        # cond_idx
        # 0 - Maximum
        # 1 - Summation
        # 2 - CALC
        # 3 - MASS



        ##############################################
        # Get cho response
        q_arr = []
        for cho_idx in range(len(cho_path)):
          q_ = sio.loadmat(cho_path[cho_idx]+stim_name_+'.mat')
          print('***** Loaded from CHO path: {}'.format(cho_path[cho_idx]+stim_name_+'.mat'))
          q = q_['ss_3d_search'] #[:,:,sl] #q = q_['ss'], #q[:,-40:]=0, #q = q - q.min(), #q = q/q.max(), #sns.heatmap(q[:,1000:], cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax4a, cbar=False, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          q_arr.append(q)
          print('** single_map: q.shape: {}'.format(q.shape))

        for cond_idx in [2,3]: #range(4): # if len(q_arr)>1:
          if cond_idx==0: 
            # Take the max at each location
            q_max = np.maximum(q_arr[0], q_arr[1])
            q_type = 'max'
          elif cond_idx==1:
            # Take the sum at each location
            q_max = (q_arr[0] + q_arr[1])/2
            q_type = 'sum'
          elif cond_idx==2:
            # CALC
            q_max = q_arr[0]
            q_type = 'calc'
          elif cond_idx==3:
            # MASS
            q_max = q_arr[1]
            q_type = 'mass'
          q_max *= tiff_bool
          q_max = gaussian_filter(q_max, sigma=[w_2, w_2, z_2])
          cho_max = q_max/np.max(q_max)
          with open(phantom_dir+q_type+'_cho_map.npy', 'wb') as f:
            np.save(f, cho_max)
          cho_indicies_arr = get_max_location(cho_max)
          print('AJ: Assigning to cho map, indicies: {}'.format(cho_indicies_arr))

        # Get fco response
        q_arr = []
        for fco_idx in range(len(fco_path)):
          q_ = sio.loadmat(fco_path[fco_idx]+stim_name_+'.mat')
          print('***** Loaded from FCO path: {}'.format(fco_path[fco_idx]+stim_name_+'.mat'))
          q = q_['ss_3d_search'] #[:,:,sl] #q = q_['ss'], #q[:,-40:]=0, #q = q - q.min(), #q = q/q.max(), #sns.heatmap(q[:,1000:], cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax4a, cbar=False, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          q_arr.append(q)

        for cond_idx in [2,3]: #range(4): # if len(q_arr)>1:
          if cond_idx==0: 
            # Take the max at each location
            q_max = np.maximum(q_arr[0], q_arr[1])
            q_type = 'max'
          elif cond_idx==1:
            # Take the sum at each location
            q_max = (q_arr[0] + q_arr[1])/2
            q_type = 'sum'
          elif cond_idx==2:
            # CALC
            q_max = q_arr[0]
            q_type = 'calc'
          elif cond_idx==3:
            # MASS
            q_max = q_arr[1]
            q_type = 'mass'

          q_type_ = q_max.dtype
          print('FCO_array {} : {}'.format(type(q_max), q_max.dtype))

          q_max *= tiff_bool
          q_max = gaussian_filter(q_max, sigma=[w_2, w_2, z_2])
          fco_max = q_max/np.max(q_max)
          with open(phantom_dir+q_type+'_fco_map.npy', 'wb') as f:
            np.save(f, fco_max)
          fco_indicies_arr = get_max_location(fco_max)
          print('AJ: Assigning to fco map, indicies: {}'.format(fco_indicies_arr))

        # Get cnn response
        q_arr = []
        for cnn_idx in range(len(cnn_path)):
          q_ = np.load(cnn_path[cnn_idx]+stim_name_+'.npy')
          print('***** Loaded from CNN path: {}, shape: {}'.format(cnn_path[cnn_idx]+stim_name_+'.npy', q_.shape))
          q = q_ #['ss_3d_search'] #[:,:,sl] #q = q_['ss'], #q[:,-40:]=0, #q = q - q.min(), #q = q/q.max(), #sns.heatmap(q[:,1000:], cmap="coolwarm", xticklabels=False, yticklabels=False, square=True, ax=ax4a, cbar=False, cbar_kws=cbar_kws, annot_kws=annot_kws)#, vmin=0.0, vmax=1.0) #cmap="YlGnBu"
          q_arr.append(q)

        for cond_idx in [2,3]: #range(4): # if len(q_arr)>1:
          if cond_idx==0: 
            # Take the max at each location
            q_max = np.maximum(q_arr[0], q_arr[1])
            q_type = 'max'
          elif cond_idx==1:
            # Take the sum at each location
            q_max = (q_arr[0] + q_arr[1])/2
            q_type = 'sum'
          elif cond_idx==2:
            # CALC
            q_max = q_arr[0]
            q_type = 'calc'
          elif cond_idx==3:
            # MASS
            q_max = q_arr[1]
            q_type = 'mass'
          q_max = q_max.astype(q_type_)
          print('CNN_array {} : {}'.format(type(q_max), q_max.dtype))
          #q_max = ndimage.convolve(q_max, dist_bool, mode='constant', cval=0.0)
          #q_max = signal.fftconvolve(q_max, dist_bool, mode='same')
          q_max = q_max[:,:-1,:]
          q_max *= tiff_bool
          q_max = gaussian_filter(q_max, sigma=[w_2, w_2, z_2])
          cnn_max = q_max/np.max(q_max)
          with open(phantom_dir+q_type+'_cnn_map.npy', 'wb') as f:
            np.save(f, cnn_max)
          cnn_indicies_arr = get_max_location(cnn_max)
          print('AJ: Assigning to cnn map, indicies: {}'.format(cnn_indicies_arr))

        #plt.close()
    return #fixation_map, time_map, cho_max, fco_max, cnn_max


save_root = '201_Visualization_maps/'
if not os.path.isdir(save_root):
  os.mkdir(save_root)

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
  

if __name__=='__main__':
  # Loop over each radiologist
  #for P_idx in np.arange(5,6): #len(participants)):
  phantom_names = []
  for P_idx in range(len(participants)):
    # Initialize required lists
    cho_mass, cho_calc, cho_noise_calc, cho_noise_mass = [], [], [], []
    npwe_mass, npwe_calc, npwe_noise = [], [], []

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
      f_name, cho_path, npwe_path, cho_th, fco_path, cnn_path = phantom_name(stim_name)
      print('stimulus_name: {}, signal_type: {},  Response: {}'.format(stim_name, f_name, response))
      print(cho_path, fco_path, cnn_path)

      ph_name = stim_name.split('\\')[-1]
      if f_name!='calc' and f_name!='mass':
        phantom_names.append(ph_name)
        print('\n\nphantom_names ({}): {}'.format(len(set(phantom_names)), set(phantom_names)))
      continue
  
  phantom_names = set(phantom_names)

  #for P_idx in range(len(participants)):
  current_cnt = 0
  for ph_name in phantom_names:
    current_cnt += 1
    if ph_name in os.listdir(root_dir+'NoLesion/'): #['Alvarado_888076.3.685169686552.20171125044315266-recon']:
      continue      

    print('**********  Current: {}/{}'.format(current_cnt, len(phantom_names)))
    z_names = []
    # Intialize the list for storing fixation information
    sc_global, lc_tm_global, lc_slice_global = [], [], []
 
    # Loop over all participants and store information if they saw this particular phantom
    for P_idx in range(len(participants)):
      # Image index seen by this radiologist
      for I_idx in np.arange(1,29): #(14,15): #(1,29): #(10,11): #(11,12):#29):#13,14):#1,29): #4): #range(11,12): #28
        #print('\nImage_idx: {}, Participant_idx: {}'.format(I_idx, P_idx))
        # Load the meta data for Radiologist and this phantom
        mat_contents = sio.loadmat('Dataset/Radiologists/'+day+'/'+participants[P_idx]+'/data'+str(I_idx)+'.mat')
        # Response by the radiologist
        response = mat_contents['resp'][0][0]['RATING'][0][0]
        # Name of the phantom shown
        stim_name = mat_contents['stimInfo'][0][0]['imgName'][0] #mat_contents['stimInfo'][0][0][8][0]
        # Corresponding location of cho and npwe responses
        f_name, cho_path, npwe_path, cho_th, fco_path, cnn_path = phantom_name(stim_name)
        #print('stimulus_name: {}, signal_type: {},  Response: {}'.format(stim_name, f_name, response))
        #print(cho_path, fco_path, cnn_path)

        ph_name_ = stim_name.split('\\')[-1]
        if f_name=='calc' or f_name=='mass' or ph_name!=ph_name_:
          #phantom_names.append(ph_name)
          #print('\n\nphantom_names ({}): {}'.format(len(set(phantom_names)), set(phantom_names)))
          continue
        #print('{}'.format(participants[P_idx]))
        z_names.append(participants[P_idx])
        
        try:
          # Extract phantom data
          tiff, cx, cy, sl, title_txt = phantom_data(stim_name, f_name)
          ratioImgScreen = [screenSize[0]/tiff.shape[1], screenSize[1]/tiff.shape[0]]

          # Extract saccade data
          sacStart, sacEnd, sacX, sacY, drill, lesionLocation = phantom_saccades(mat_contents)

          # Process drilling movements
          changeTime, allTimes = process_drills(drill, sacEnd)
      
          # Process eye-movements
          sc, lc_tm, lc_slice = process_eyemovements(sacEnd, sacX, sacY, drill, lesionLocation, ratioImgScreen, allTimes)

          print('AJ: sc.shape: {}, len(lc_tm): {}, len(lc_slice): {} ({}-{})'.format(sc.shape, len(lc_tm), len(lc_slice), np.min(lc_slice), np.max(lc_slice)))
          cho_path_global = cho_path
          npwe_path_global = npwe_path
          fco_path_global = fco_path
          cnn_path_global = cnn_path
      

          sc_global.append(sc)
          lc_tm_global.extend(lc_tm)
          lc_slice_global.extend(lc_slice)
          stim_name_global = stim_name
          
          phantom_root = root_dir + f_name + '/'
          try:
            os.mkdir(phantom_root)
          except:
            pass
          phantom_dir = phantom_root + stim_name.split('\\')[-1] + '/'
          try:
            os.mkdir(phantom_dir)
          except:
            pass 
          
          #'''
          tiff_bool = tiff[:,1000:]/16384.0
          tiff_kernel = (1/400)*np.ones((20,20,1))
          tiff_bool = signal.fftconvolve(tiff_bool, tiff_kernel, mode='same')
          tiff_bool = tiff_bool>0.4272
          tiff_bool[:,-80:]=0



          Generate_human_maps_SMALL(stim_name, sc, lc_tm, lc_slice, ratioImgScreen, mat_contents, cx, cy, sl, f_name, title_txt, response, participants[P_idx], phantom_dir, tiff_bool)
          #'''
        except:
          pass
        #continue
        #break
        # CHO and Density
        #cho_val = plot_fixations_cho_density(stim_name, sc, lc_tm, lc_slice, ratioImgScreen, mat_contents, cx, cy, f_name, title_txt, response, participants[P_idx], phantom_dir, cho_th)
    print('{}: {}'.format(ph_name, z_names))
    
    if True: #len(cho_path)==1:
      # All fixations are collected, create the maps now 
      sc_global = np.concatenate(sc_global, axis=0)
      f_name = 'NoLesion'
      print('AJ*****: sc_global.shape: {}, len(lc_tm_global): {}, len(lc_slice_global): {} ({}-{})'.format(sc_global.shape, len(lc_tm_global), len(lc_slice_global), np.min(lc_slice_global), np.max(lc_slice_global)))
      #break

      # Mask generation
      tiff_bool = tiff[:,1000:]/16384.0
      tiff_kernel = (1/400)*np.ones((20,20,1))
      tiff_bool = signal.fftconvolve(tiff_bool, tiff_kernel, mode='same')
      tiff_bool = tiff_bool>0.4272
      tiff_bool[:,-80:]=0
      tiff_area = np.sum(tiff_bool)/(2048*792*64)
      print('****** AJ: Name: {}, tiff.shape: {}, percentage_area: {}'.format(ph_name, tiff_bool.shape, tiff_area))

      # Human maps
      print(stim_name_global)
      #fixation_map, time_map, cho_map, fco_map, cnn_map = Generate_human_maps(stim_name_global, sc_global, lc_tm_global, lc_slice_global, ratioImgScreen, mat_contents, cx, cy, sl, f_name, title_txt, response, 'combined', phantom_dir, cho_path_global, fco_path_global, cnn_path_global, tiff_bool)
      Generate_human_maps(stim_name_global, sc_global, lc_tm_global, lc_slice_global, ratioImgScreen, mat_contents, cx, cy, sl, f_name, title_txt, response, 'combined', phantom_dir, cho_path_global, fco_path_global, cnn_path_global, tiff_bool)
      #print('fixation_map.shape: {}, max: {}, min: {}'.format(fixation_map.shape, np.max(fixation_map), np.min(fixation_map)))
      #print('time_map.shape: {}, max: {}, min: {}'.format(time_map.shape, np.max(time_map), np.min(time_map)))
