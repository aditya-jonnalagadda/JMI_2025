% Generating the weights for the slices
flag_sigsize = '2D';
data_dir = 'Train_data_NEW/';

for model_num = 0:1
    if model_num==0
        flag_modelname='CHO';
    else
        flag_modelname='FCO';
    end
    %mkdir(['Results/' flag_modelname]);
    
    for mtype_idx = 0:1
        if (mtype_idx==0)
            sig_type = 'CALC/';
            sig_name = 'calc';
            spvoi = load([data_dir '3D_Crops_calc_present.mat']).all_crops;
            savoi = load([data_dir '3D_Crops_calc_absent.mat']).all_crops;
        else
            sig_type = 'MASS/';
            sig_name = 'mass';
            spvoi = load([data_dir '3D_Crops_mass_present.mat']).all_crops;
            savoi = load([data_dir '3D_Crops_mass_absent.mat']).all_crops;
        end
        
        nsa = size(savoi,4); %number of SA cases
        nsp = size(spvoi,4); %number of SP cases
        
        ntrain = 450; %40;
        id_sa_tr=[1:nsa];
        id_sp_tr=[1:nsp];
        id_sa_test=[1:nsa];
        id_sp_test=[1:nsp];
        
        trimg_sa=savoi(:,:,:,id_sa_tr);
        trimg_sp=spvoi(:,:,:,id_sp_tr);
        ntr_sa = size(trimg_sa,4);
        ntr_sp = size(trimg_sp,4);
        
        
        
        single_template_3d = zeros(10201,0);
        for slice_idx=1:17
            template = load(['templates/Slice_' num2str(slice_idx) '_' flag_sigsize '_' flag_modelname '_' sig_type(1:end-1) '.mat']).template;
            comb_temp = squeeze(template.ch*template.w');
            if isempty(single_template_3d)
                single_template_3d = comb_temp;
            else
                single_template_3d = cat(2, single_template_3d, comb_temp);
            end
        end
        
        % Now we have the 3d template which is 10201x17
        % Do the steps similar to the generation of CHO channel weights
        nxny=10201;%nx*ny;
        ch = single_template_3d;
        nch=size(single_template_3d,2);
        tr_sa_ch = zeros(nch, ntr_sa);
        tr_sp_ch = zeros(nch, ntr_sp);
        for i=1:ntr_sa
            for j=1:17
                tr_sa_ch(j,i) = reshape(trimg_sa(:,:,j,i), 1,nxny)*ch(:,j);
            end
        end
        for i=1:ntr_sp
            for j=1:17
                tr_sp_ch(j,i) = reshape(trimg_sp(:,:,j,i), 1,nxny)*ch(:,j);
            end
        end
        s_ch = mean(tr_sp_ch,2) - mean(tr_sa_ch,2);
        k_sa = cov(tr_sa_ch');
        k_sp = cov(tr_sp_ch');
        k = (k_sa+k_sp)/2;
        w = s_ch(:)'*pinv(k); %this is the hotelling template
        
        if mtype_idx==0
            if model_num==0
                save(['templates/Slice_weights_2D_CHO_CALC.mat'], 'w');
            else
                save(['templates/Slice_weights_2D_FCO_CALC.mat'], 'w');
            end
        else
            if model_num==0
                save(['templates/Slice_weights_2D_CHO_MASS.mat'], 'w');
            else
                save(['templates/Slice_weights_2D_FCO_MASS.mat'], 'w');
            end
        end
        
    end
end
