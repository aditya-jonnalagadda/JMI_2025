import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti

import scipy.io as sio

flag_crop = False #True

if __name__ == '__main__':
    """
    nnU-Net was originally built for 3D images. It is also strongest when applied to 3D segmentation problems because a 
    large proportion of its design choices were built with 3D in mind. Also note that many 2D segmentation problems, 
    especially in the non-biomedical domain, may benefit from pretrained network architectures which nnU-Net does not
    support.
    Still, there is certainly a need for an out of the box segmentation solution for 2D segmentation problems. And 
    also on 2D segmentation tasks nnU-Net cam perform extremely well! We have, for example, won a 2D task in the cell 
    tracking challenge with nnU-Net (see our Nature Methods paper) and we have also successfully applied nnU-Net to 
    histopathological segmentation problems. 
    Working with 2D data in nnU-Net requires a small workaround in the creation of the dataset. Essentially, all images 
    must be converted to pseudo 3D images (so an image with shape (X, Y) needs to be converted to an image with shape 
    (1, X, Y). The resulting image must be saved in nifti format. Hereby it is important to set the spacing of the 
    first axis (the one with shape 1) to a value larger than the others. If you are working with niftis anyways, then 
    doing this should be easy for you. This example here is intended for demonstrating how nnU-Net can be used with 
    'regular' 2D images. We selected the massachusetts road segmentation dataset for this because it can be obtained 
    easily, it comes with a good amount of training cases but is still not too large to be difficult to handle.
    """

    # download dataset from https://www.kaggle.com/insaff/massachusetts-roads-dataset
    # extract the zip file, then set the following path according to your system:
    base_root = '/mnt/edward/data/adityaj/nnUNet_2022/Step_0_data_formatting/N_MASS_folder_3D_New/'
    base = base_root + 'all_data/'

    # Load gts
    gts = sio.loadmat(base_root + 'signal_gts.mat')
    signal_gts = gts['signal_gts']

    # this folder should have the training and testing subfolders

    # now start the conversion to nnU-Net:
    task_name = 'Task812_MASS2D'
    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)



    train_startidx = 1
    train_stopidx = 501
    test_startidx = 501
    test_stopidx = 551

    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    #labels_dir_tr = join(base, 'training', 'output')
    #images_dir_tr = join(base, 'training', 'input')
    #training_cases = subfiles(labels_dir_tr, suffix='.png', join=False)
    #for t in training_cases:


    for idx in range(train_startidx, train_stopidx):
        if flag_crop:
            pix_side = 380
            pix_side2 = np.int(pix_side/2)
            row = signal_gts[idx-1][0]-1
            col = signal_gts[idx-1][1]-1
            # Find boundaries
            rb = col+pix_side2
            if rb>793:
                rb = 793
            lb = rb-pix_side
            bb = row+pix_side2
            if bb>2048:
                bb = 2048
            tb = bb-pix_side
        
        slice_idx = signal_gts[idx-1][2]-1 #idx-1 for signal ordering, -1 for slice ordering
        mat_contents = sio.loadmat(base + 'Signal_'+str(idx)+'/images/Signal_'+ str(idx)+'.mat')
        if flag_crop:
            input_image_file = mat_contents['I'][tb:bb,lb:rb,slice_idx]
        else:
            input_image_file = mat_contents['I'][:,:,slice_idx]
        mat_contents_seg = sio.loadmat(base + 'Signal_'+str(idx)+'/masks/mask.mat')
        if flag_crop:
            input_segmentation_file = mat_contents_seg['Mask'][tb:bb,lb:rb,slice_idx]
        else:
            input_segmentation_file = mat_contents_seg['Mask'][:,:,slice_idx]
        zero_segmentation_file = np.zeros_like(input_segmentation_file)
        print('idx: {}, Signal Mask sum: {}'.format(idx, np.sum(input_segmentation_file)))

        unique_name = 'Signal_'+str(idx) #t[:-4]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        #input_segmentation_file = join(labels_dir_tr, t)
        #input_image_file = join(images_dir_tr, t)

        output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        print('max Mask: {}'.format(np.max(input_segmentation_file)))

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 1).astype(int))

        
        slice_idx = signal_gts[idx-1][2] #idx-1 for signal ordering, -1 for slice ordering
        mat_contents = sio.loadmat(base + 'Signal_'+str(idx)+'/images/Signal_'+ str(idx)+'.mat')
        if flag_crop:
            input_image_file = mat_contents['I'][tb:bb,lb:rb,slice_idx]
        else:
            input_image_file = mat_contents['I'][:,:,slice_idx]
        mat_contents_seg = sio.loadmat(base + 'Signal_'+str(idx)+'/masks/mask.mat')
        if flag_crop:
            input_segmentation_file = mat_contents_seg['Mask'][tb:bb,lb:rb,slice_idx]
        else:
            input_segmentation_file = mat_contents_seg['Mask'][:,:,slice_idx]
        zero_segmentation_file = np.zeros_like(input_segmentation_file)
        print('Signal Mask sum: {}'.format(np.sum(input_segmentation_file)))

        unique_name = 'Signal_'+str(1000+idx) #t[:-4]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        #input_segmentation_file = join(labels_dir_tr, t)
        #input_image_file = join(images_dir_tr, t)

        output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        print('max Mask: {}'.format(np.max(input_segmentation_file)))

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 1).astype(int))

        
        slice_idx = signal_gts[idx-1][2]+1 #idx-1 for signal ordering, -1 for slice ordering
        mat_contents = sio.loadmat(base + 'Signal_'+str(idx)+'/images/Signal_'+ str(idx)+'.mat')
        if flag_crop:
            input_image_file = mat_contents['I'][tb:bb,lb:rb,slice_idx]
        else:
            input_image_file = mat_contents['I'][:,:,slice_idx]
        mat_contents_seg = sio.loadmat(base + 'Signal_'+str(idx)+'/masks/mask.mat')
        if flag_crop:
            input_segmentation_file = mat_contents_seg['Mask'][tb:bb,lb:rb,slice_idx]
        else:
            input_segmentation_file = mat_contents_seg['Mask'][:,:,slice_idx]
        zero_segmentation_file = np.zeros_like(input_segmentation_file)
        print('Signal Mask sum: {}'.format(np.sum(input_segmentation_file)))

        unique_name = 'Signal_'+str(2000+idx) #t[:-4]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        #input_segmentation_file = join(labels_dir_tr, t)
        #input_image_file = join(images_dir_tr, t)

        output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        print('max Mask: {}'.format(np.max(input_segmentation_file)))

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 1).astype(int))

        'Noise'
        '''
        mat_contents = sio.loadmat(base + 'Noise_'+str(idx)+'/images/Noise_'+ str(idx)+'.mat')
        input_image_file = mat_contents['I'][tb:bb,lb:rb,slice_idx]
        input_segmentation_file = zero_segmentation_file

        print('Noise Mask sum: {}'.format(np.sum(input_segmentation_file)))

        unique_name = 'Noise_'+str(idx) #t[:-4]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        #input_segmentation_file = join(labels_dir_tr, t)
        #input_image_file = join(images_dir_tr, t)

        output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 1).astype(int))
        '''
    # now do the same for the test set
    #labels_dir_ts = join(base, 'testing', 'output')
    #images_dir_ts = join(base, 'testing', 'input')
    #testing_cases = subfiles(labels_dir_ts, suffix='.png', join=False)
    #for ts in testing_cases:


    for idx in range(test_startidx, test_stopidx):
        slice_idx = signal_gts[idx-1][2]-1 #idx-1 for signal ordering, -1 for slice ordering
        mat_contents = sio.loadmat(base + 'Signal_'+str(idx)+'/images/Signal_'+ str(idx)+'.mat')
        input_image_file = mat_contents['I'][:,:,slice_idx]
        mat_contents_seg = sio.loadmat(base + 'Signal_'+str(idx)+'/masks/mask.mat')
        input_segmentation_file = mat_contents_seg['Mask'][:,:,slice_idx]
        zero_segmentation_file = np.zeros_like(input_segmentation_file)

        unique_name = 'Signal_'+str(idx) #ts[:-4]
        #input_segmentation_file = join(labels_dir_ts, ts)
        #input_image_file = join(images_dir_ts, ts)

        output_image_file = join(target_imagesTs, unique_name)
        output_seg_file = join(target_labelsTs, unique_name)

        #print('Test: Signal: input_image_file.shape: {}, input_segmentation_file.shape: {}'.format(input_image_file.shape, input_segmentation_file.shape))

        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 1).astype(int))

        # Noise
        '''
        mat_contents = sio.loadmat(base + 'Noise_'+str(idx)+'/images/Noise_'+ str(idx)+'.mat')
        input_image_file = mat_contents['I'][:,:,slice_idx]
        
        input_segmentation_file = zero_segmentation_file


        unique_name = 'Noise_'+str(idx) #ts[:-4]
        #input_segmentation_file = join(labels_dir_ts, ts)
        #input_image_file = join(images_dir_ts, ts)

        output_image_file = join(target_imagesTs, unique_name)
        output_seg_file = join(target_labelsTs, unique_name)

        #print('Test: Noise: input_image_file.shape: {}, input_segmentation_file.shape: {}'.format(input_image_file.shape, input_segmentation_file.shape))

        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 1).astype(int))
        '''
    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Phantom',),
                          labels={0: 'background', 1: 'tumor'}, dataset_name=task_name, license='hands off!')

    """
    once this is completed, you can use the dataset like any other nnU-Net dataset. Note that since this is a 2D
    dataset there is no need to run preprocessing for 3D U-Nets. You should therefore run the 
    `nnUNet_plan_and_preprocess` command like this:
    
    > nnUNet_plan_and_preprocess -t 701 -pl3d None
    Not required?    nnUNet_plan_and_preprocess -t 701 --verify_dataset_integrity
 
    once that is completed, you can run the trainings as follows:
    > nnUNet_train 2d nnUNetTrainerV2 701 FOLD
    
    (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)
    
    there is no need to run nnUNet_find_best_configuration because there is only one model to choose from.
    Note that without running nnUNet_find_best_configuration, nnU-Net will not have determined a postprocessing
    for the whole cross-validation. Spoiler: it will determine not to run postprocessing anyways. If you are using
    a different 2D dataset, you can make nnU-Net determine the postprocessing by using the
    `nnUNet_determine_postprocessing` command
    """
