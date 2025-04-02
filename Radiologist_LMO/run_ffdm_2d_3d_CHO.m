%Read in ROIs
%load ffdm_saroi;  %2d signal-absent (SA) ROIs
%load ffdm_sproi;  %2d signal-present (SP) ROIs
%ntrain = 100;

% CHANGE cyc/deg for both types of signals
clear all

for flag_calc = 0:1
    
    
    
    try
        mkdir('templates');
    catch
        disp('Folder exists!!');
    end
    
    
    data_dir = 'Train_data_NEW/'; %'Train_data_Sep_12/';
    if flag_calc
        spvoi = load([data_dir '3D_Crops_calc_present.mat']).all_crops;
        savoi = load([data_dir '3D_Crops_calc_absent.mat']).all_crops;
    else
        spvoi = load([data_dir '3D_Crops_mass_present.mat']).all_crops;
        savoi = load([data_dir '3D_Crops_mass_absent.mat']).all_crops;
    end
    
    for slice_idx = 1:17 %; % 1-17
        
        sproi = squeeze(spvoi(:,:,slice_idx,:));
        saroi = squeeze(savoi(:,:,slice_idx,:));
        
        nsa = size(saroi,3); %number of SA cases
        nsp = size(sproi,3); %number of SP cases
        
        id_sa_tr=[1:nsa];%[1:ntrain];
        id_sp_tr=[1:nsp];%[1:ntrain];
        id_sa_test=[1:nsa];%[ntrain+1:nsa];
        id_sp_test=[1:nsp];%[ntrain+1:nsp];
        
        %run CHO by setting the last parameter to 0
        [ch1, w1, snr1, AUC1, t_sp1, t_sa1, chimg1,tplimg1,meanSP1,meanSA1,meanSig1, k_ch1]=conv_LG_CHO_2d(saroi(:,:,id_sa_tr), sproi(:,:,id_sp_tr), saroi(:,:,id_sa_test), sproi(:,:,id_sp_test),30,5,0, flag_calc);
        %run convolutional CHO by setting the last parameter to 1
        [ch2, w2, snr2, AUC2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_2d(saroi(:,:,id_sa_tr), sproi(:,:,id_sp_tr), saroi(:,:,id_sa_test), sproi(:,:,id_sp_test),30,5,1, flag_calc);
        
        
        disp(['CHO: AUC: ' num2str(AUC1) ' (Slice: ' num2str(slice_idx) ')']);
        disp(['FCO: AUC: ' num2str(AUC2) ' (Slice: ' num2str(slice_idx) ')']);
        
        
        
        if flag_calc
            template.ch = ch1;
            template.w = w1;
            template.snr = snr1;
            template.AUC = AUC1;
            save(['templates/Slice_' num2str(slice_idx) '_2D_CHO_CALC.mat'], 'template');
            template.ch = ch2;
            template.w = w2;
            template.snr = snr2;
            template.AUC = AUC2;
            save(['templates/Slice_' num2str(slice_idx) '_2D_FCO_CALC.mat'], 'template');
        else
            template.ch = ch1;
            template.w = w1;
            template.snr = snr1;
            template.AUC = AUC1;
            save(['templates/Slice_' num2str(slice_idx) '_2D_CHO_MASS.mat'], 'template');
            template.ch = ch2;
            template.w = w2;
            template.snr = snr2;
            template.AUC = AUC2;
            save(['templates/Slice_' num2str(slice_idx) '_2D_FCO_MASS.mat'], 'template');
        end
        
        %subplot(2,2,1);imshow(meanSP1,[]);title('CHO-CALC signal template'); subplot(2,2,2); imshow(meanSP2,[]); title('FCO-CALC signal template');subplot(2,2,3);imshow(meanSA1,[]);title('CHO-CALC noise template'); subplot(2,2,4); imshow(meanSA2,[]); title('FCO-CALC noise template');
        close all;
    end
    
end