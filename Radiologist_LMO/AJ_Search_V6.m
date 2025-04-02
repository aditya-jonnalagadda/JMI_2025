%%
clearvars

root_dir = './';%'/media/aditya/VIU_Lacie/Aditya_temp/Winter_2021/Radiologists_ModelObservers/';

%addpath([root_dir 'templatesPH/Tools']);
%addpath([root_dir 'templatesPH']);

%mkdir('Results/CHO/2D/');
%mkdir('Results/CHO/2D/CALC/Signal/');
%mkdir('Results/CHO/2D/CALC/Noise/');

% Test on the 2018 phantoms to generate the results in SPIE paper
%flag_modelname = 'CHO'; %'FCO'; %'CHO';
flag_sigsize = '2D';



mkdir('Results/');

mkdir('Results/Images/');
mkdir('Results/Files/');


load('deterministic_var.mat');


load([root_dir '2017_testdata.mat']);
calc_names = calc_testnames; mass_names = mass_testnames; noise_names = noise_testnames;

for model_num = 0:1
    if model_num==0
        flag_modelname='CHO';
    else
        flag_modelname='FCO';
    end
    mkdir(['Results/' flag_modelname]);
    
    for mtype_idx = 0:1
        if (mtype_idx==0)
            sig_type = 'CALC/';
            sig_name = 'calc';
        else
            sig_type = 'MASS/';
            sig_name = 'mass';
        end
        
        % Load template
        %template = load(['templates/Slice_10_' flag_sigsize '_' flag_modelname '_' sig_type(1:end-1) '.mat']).template;
        %template_3d = load(['templates_DOG/3D_' flag_modelname '_' sig_type(1:end-1) '.mat']).template;
        central_slice = 9;
        template = load(['templates/Slice_' num2str(central_slice) '_' flag_sigsize '_' flag_modelname '_' sig_type(1:end-1) '.mat']).template;
        slice_weights = squeeze(load(['templates/Slice_weights_' flag_sigsize '_' flag_modelname '_' sig_type(1:end-1) '.mat']).w);
        single_template_2d = reshape(squeeze(template.ch*template.w'), 101, 101);
        % Construct the 3D template from 2D templates
        single_template_3d = zeros(10201,0);
        for slice_idx=1:17
            template = load(['templates/Slice_' num2str(slice_idx) '_' flag_sigsize '_' flag_modelname '_' sig_type(1:end-1) '.mat']).template;
            comb_temp = slice_weights(slice_idx)*squeeze(template.ch*template.w');
            %%comb_temp = squeeze(template.ch*template.w');
            if isempty(single_template_3d)
                single_template_3d = comb_temp;
            else
                single_template_3d = cat(2, single_template_3d, comb_temp);
            end
        end
        single_template_3d = reshape(single_template_3d, 101, 101, 17);
        
        %  textprogressbar('Samples:');
        nch = 1;
        ch_nch = size(template.ch,2); %10; %80;
        
        % Compute the fft of channel 2d
        crop_pix = 1000; ch_fft_dict = cell(nch,1); channel_tmp_global=0;
        %for ch_idx=1:ch_nch
        %    %channel_tmp =  reshape(template.ch(:,ch_idx),101,101);
        %    channel_tmp = flip(flip(single_template_2d),2);
        %    channel_tmp = channel_tmp*template.w(ch_idx);
        %    channel_tmp_global = channel_tmp_global+channel_tmp;
        %end
        channel_tmp_global = flip(flip(single_template_2d),2);
        %pad before !!!!!!!!!!!!!!!!!!!!!!!!
        %continue
        ch_fft_2d_cropped = padarray(channel_tmp_global, [51, 51], 0, 'both');
        ch_fft_2d_cropped = fftshift(fftn(ch_fft_2d_cropped));
        %channel_tmp_global = padarray(channel_tmp_global, [1024, 524], 0, 'both');
        %channel_tmp_global = padarray(channel_tmp_global, [973, 345], 0, 'pre');
        %channel_tmp_global = padarray(channel_tmp_global, [974, 346], 0, 'post');
        % DO I NEED fftshift before applying fft???????
        %fftn(fftshift(channel_tmp_global));%,[2304, 2048-crop_pix]);%[2048,1792]);
        ch_fft_ = padarray(channel_tmp_global, [973, 345], 0, 'pre');
        ch_fft_ = padarray(ch_fft_, [974, 346], 0, 'post');
        ch_fft_ = fftshift(fftn(ch_fft_));
        ch_fft_dict{1} = ch_fft_;
        % Compute the fft of channel 3d
        ch_3d_fft_dict = cell(nch,1); channel_tmp_global=0;
        if 1
            % for ch_idx=1:ch_nch
            %     %disp(['channel idx: ' num2str(ch_idx)]);
            %     %tic;
            %     channel_tmp =  reshape(template_3d.ch(:,ch_idx),101,101,17);
            %     channel_tmp = flip(flip(flip(channel_tmp),2),3);
            %     channel_tmp = channel_tmp*template_3d.w(ch_idx);
            %     channel_tmp_global = channel_tmp_global + channel_tmp;
            %     %toc;
            % end
            channel_tmp_global = flip(flip(flip(single_template_3d),2),3);
            ch_fft_3d_cropped = padarray(channel_tmp_global, [51, 51, 9], 0, 'both');
            ch_fft_3d_cropped = fftshift(fftn(ch_fft_3d_cropped));
            %channel_tmp_global_3d = channel_tmp_global;
            %channel_tmp_global = padarray(channel_tmp_global, [973, 345, 23], 0, 'pre');
            %channel_tmp_global = padarray(channel_tmp_global, [974, 346, 24], 0, 'post');
            %,[2304, 2048-crop_pix, 82]);%[2048,1792]);
            %ch_fft_ = padarray(ch_fft_, [973, 345, 23], 0, 'pre');
            %ch_fft_ = padarray(ch_fft_, [974, 346, 24], 0, 'post');
            ch_fft_ = padarray(channel_tmp_global, [973, 345, 31], 0, 'pre');
            ch_fft_ = padarray(ch_fft_, [974, 346, 32], 0, 'post');
            ch_fft_ = fftshift(fftn(ch_fft_));
            ch_3d_fft_dict{1} = ch_fft_;
        end
        
        for flag_Signal = 0:1
            disp(['flag_Signal: ' num2str(flag_Signal)]);
            if flag_Signal
                sn_type = 'Signal/';
                sn_name = 'present';
            else
                sn_type = 'Noise/';
                sn_name = 'absent';
            end
            
            if flag_Signal
                all_crops = zeros(101, 101, 14); all_cnt = 1;
                all_crops_3d = zeros(101, 101, 17, 14); all_cnt_3d = 1;
            else
                all_crops = zeros(101, 101, 28); all_cnt = 1;
                all_crops_3d = zeros(101, 101, 17, 28); all_cnt_3d = 1;
            end
            
            
            if flag_Signal
                if (mtype_idx == 0)
                    phDir = 'AJ_Combined/calc';
                    names_list = calc_names;
                elseif (mtype_idx==1)
                    phDir = 'AJ_Combined/mass';
                    names_list = mass_names;
                end
            else
                phDir = 'AJ_Combined/NoLesion';
                names_list = noise_names;
            end
            
            
            dir_list = dir(phDir);
            dir_list(ismember({dir_list.name}, {'.','..'})) = [];
            
            
            
            %%
            %w=cell(2,numel(eccentricities));
            %w2D=cell(2,numel(eccentricities));
            %r3D=cell(2,numel(eccentricities));
            %r2D=cell(2,numel(eccentricities));
            resp_list_LKE = []; resp_list_search = [];
            resp_list_3d=[]; resp_list_3d_search = [];
            val_list = [];
            
            
            
            for n=1:numel(dir_list)
                if mod(n,10)==0
                    disp([n numel(dir_list)]);
                end
                for s=1 %0:1
                    if 1
                        
                        try
                            locations=load([phDir '/' dir_list(n).name '/locations.txt']);
                            signal_presence = 1;
                        catch
                            signal_presence = 0;
                        end
                        
                        
                        % When working with 2017 dataset, proceed only if the
                        % current names is only in the list of expected names
                        if ~any(strcmp(dir_list(n).name, names_list))
                            continue;
                        end
                        
                        ph=load([phDir '/' dir_list(n).name '/phantom.mat']);
                        ph.tiff = double(ph.tiff(:,(crop_pix+1):end,:))/16384;%8192;
                        
                        w_2=50;
                        if flag_Signal %try
                            locations=load([phDir '/' dir_list(n).name '/locations.txt']);
                            ph2D=ph.tiff(:,:,locations(2));
                            signal_presence = 1;
                            X = floor(locations(1)*10)-crop_pix;
                            Y = floor(locations(3)*10);
                            Z = floor(locations(2));
                            num_loop = 1;
                            X_sig = X; Y_sig = Y; Z_sig = Z;
                        else %catch
                            signal_presence = 0;
                            X = deterministic_var(1024,1); %randi([(1001+w_2-crop_pix) (1792-w_2-crop_pix)]);
                            Y = deterministic_var(1024,2); %randi([(w_2+501) (1500-w_2)]);
                            Z = deterministic_var(1024,3); %randi([9 52]); %32
                            ph2D = ph.tiff(:,:,Z);
                            %crop = tiff((Y-w_2):(Y+w_2), (X-w_2):(X+w_2), (Z-8):(Z+8));
                            num_loop = 256;
                        end
                        
                        flag_plot = 0;
                        
                        if flag_plot
                            if flag_Signal
                                z_slice = Z;
                            else
                                z_slice = 32;
                            end
                            ph2D = ph.tiff(:,:,z_slice);
                            %                             subplot(1,4,1);
                            %                             imshow(ph2D, []); hold on;
                            %                             if flag_Signal
                            %                                 pos = [X-50, Y-50, 100, 100];
                            %                                 rectangle('Position',pos,'EdgeColor','g','LineWidth',3);
                            %                             end
                            %                             Q = split(dir_list(n).name,'.');
                            %                             title({Q{4,1}; [' signal: ' num2str(flag_Signal)]; ['Slice: ' num2str(z_slice)]});
                        end
                        
                        
                        %, [2304, 2048-crop_pix]);
                        
                        
                        %ph2D=double(ph2D); %/256.0;
                        %ph.tiff=double(ph.tiff); %/256.0;% Try dividing with 256.0 and operate on double datatype if required
                        ss_search = zeros(2,2);
                        ss_3d_search = zeros(2,2);
                            
                        for ecci=1
                            
                            
                            % Post-processing
                            a = repmat(imopen(ph2D>0.4272,strel('disk',20)),1,1,1);
                            a(:,end-80:end)=0;
                            
                            while num_loop>0
                                ph2D = ph.tiff(:,:,Z);
                                
                                crop = double(ph2D((Y-w_2):(Y+w_2), (X-w_2):(X+w_2)));
                                all_crops(:,:,all_cnt) = crop; all_cnt = all_cnt+1;
                                
                                crop_3d = double(ph.tiff((Y-w_2):(Y+w_2), (X-w_2):(X+w_2), (Z-8):(Z+8)));
                                all_crops_3d(:,:,:,all_cnt_3d) = crop_3d; all_cnt_3d = all_cnt_3d+1;
                                
                                %ss=template.w(:)'*(reshape(crop, 1, 101*101)*template.ch)';
                                %ss_3d=template_3d.w(:)'*(reshape(crop_3d, 1, 101*101*17)*template_3d.ch)';
                                
                                
                                
                                % 2D LKE
                                crop = padarray(crop, [51, 51], 0, 'both');
                                crop_fft = fftshift(fftn(crop));
                                response = abs(ifftshift(ifftn(ch_fft_2d_cropped.*crop_fft)));
                                ss_LKE = response(102,102);
                                
                                % 3D LKE
                                crop_3d = padarray(crop_3d, [51, 51, 9], 0, 'both');
                                crop_fft_3d = fftshift(fftn(crop_3d));
                                response = abs(ifftshift(ifftn(ch_fft_3d_cropped.*crop_fft_3d)));
                                ss_3d_LKE = response(102,102,18);
                                
                                resp_list_LKE = [resp_list_LKE; max(ss_LKE(:))];
                                resp_list_3d = [resp_list_3d; max(ss_3d_LKE(:))];
                                
                                % 2D Search
                                if 1
                                    if num_loop<65
                                        if flag_Signal
                                            ph2D = ph.tiff(:,:,Z);
                                            Z_slice = Z;
                                        else
                                            ph2D = ph.tiff(:,:,num_loop);
                                            Z_slice = num_loop;
                                        end
                                        sa2D=fftshift(fftn(ph2D));
                                        ss_search = abs(ifftshift(ifftn(sa2D.*ch_fft_dict{1})));
                                        ss_search = ss_search.*single(a);
                                        resp_list_search = [resp_list_search; max(ss_search(:))];
                                        
                                        if num_loop==32 || flag_Signal
                                            if flag_plot
                                                subplot(1,4,1);
                                                imshow(ph2D, []); hold on;
                                                if flag_Signal
                                                    pos = [X_sig-50, Y_sig-50, 100, 100];
                                                    rectangle('Position',pos,'EdgeColor','g','LineWidth',1);
                                                    z_slice = Z_sig;
                                                else
                                                    z_slice = num_loop;
                                                end
                                                Q = split(dir_list(n).name,'.');
                                                title({Q{4,1}; [' signal: ' num2str(flag_Signal)]; ['Slice: ' num2str(z_slice)]});
                                                
                                                subplot(1,4,2);
                                                [max_num,max_idx] = max(ss_search(:));
                                                [Y_, X_]=ind2sub(size(ss_search),max_idx);
                                                imshow(ss_search, []); hold on;
                                                pos = [X_-50, Y_-50, 100, 100];
                                                rectangle('Position',pos,'EdgeColor','r','LineWidth',1);
                                                title({'2D Search'; ['Max value: ' num2str(floor(max_num*10000)/10000)]; ['Slice: ' num2str(z_slice)]});                                                
                                           end
                                        end
                                    end
                                else
                                    resp_list_search = [resp_list_search; max(ss_search(:))];
                                end
                                
                                X = deterministic_var(num_loop,1); %randi([(1001+w_2-crop_pix) (1792-w_2-crop_pix)]);
                                Y = deterministic_var(num_loop,2); %randi([(w_2+501) (1500-w_2)]);
                                Z = deterministic_var(num_loop,3); %randi([9 52]);
                                
                                num_loop = num_loop - 1;
                            end
                            
                            
                            
                            

                            if 1

                                


                                % 3D Search
                                Img = ph.tiff;
                                sa = padarray(Img, [0, 0, 8], 0, 'both');
                                sa = fftshift(fftn(sa));
                                ss_3d_search = abs(ifftshift(ifftn(sa.*ch_3d_fft_dict{1})));
                                % Post-Processing
                                a = repmat(imopen(ph.tiff>0.4272,strel('disk',20)),1,1,1);
                                a(:,end-80:end,:)=0;
                                ss_3d_search = ss_3d_search(:,:,9:72).*single(a);
                                save(['Results/Files/' flag_modelname '_' sig_name '_' sn_name '_' num2str(n) '_' dir_list(n).name '.mat'], 'ss_3d_search');
                                
%                                 ss_3d_search_ = ss_3d_search(:,:,Z);
%                                 subplot(1,4,3);
%                                 [max_num,max_idx] = max(ss_3d_search_(:));
%                                 [Y_, X_]=ind2sub(size(ss_3d_search_),max_idx);
%                                 imshow(ss_3d_search_, []); hold on;
%                                 pos = [X_-50, Y_-50, 100, 100];
%                                 rectangle('Position',pos,'EdgeColor','r','LineWidth',3);
%                                 title({'3D response'; ['Max value: ' num2str(floor(max_num*10000)/10000)]; ['Slice: ' num2str(Z)]});
%
                                if flag_plot
                                    % Get the best slice
                                    sl_max = zeros(5,1);
                                    for sidx=1:64
                                        ss_3d_search_ = ss_3d_search(:,:,sidx);
                                        sl_max(sidx) = max(ss_3d_search_(:));
                                    end
                                    [a,b] = sort(sl_max, 'descend');
                                    Z = b(1);
                                    ss_3d_search_ = ss_3d_search(:,:,Z);
                                    subplot(1,4,4);
                                    [max_num,max_idx] = max(ss_3d_search_(:));
                                    [Y_, X_]=ind2sub(size(ss_3d_search_),max_idx);
                                    imshow(ss_3d_search_, []); hold on;
                                    pos = [X_-50, Y_-50, 100, 100];
                                    rectangle('Position',pos,'EdgeColor','r','LineWidth',1);
                                    title({'3D Search'; ['Max value: ' num2str(floor(max(ss_3d_search(:))*10000)/10000)]; ['Max slice: ' num2str(Z)]});
                                    
                                    ph2D = ph.tiff(:,:,Z);
                                    subplot(1,4,3);
                                    imshow(ph2D, []); 
                                    Q = split(dir_list(n).name,'.');
                                    title({ ['Slice: ' num2str(Z)]});
                                end

                                %saveas(gcf, ['Results/Images/' flag_modelname '_' sig_name '_' sn_name '_' num2str(n) '.png']);
                                f = gcf;
                                exportgraphics(f,['Results/Images/' flag_modelname '_' sig_name '_' sn_name '_' num2str(n) '.png'],'Resolution',300)
                                close(gcf);
%                                 
                                % figure;
                                % cnt = 1;
                                % for sidx = 35:42 %28:35
                                %     ph2D = ph.tiff(:,:,sidx);
                                %     sa2D=fftshift(fftn(ph2D));
                                %   ss_search = abs(ifftshift(ifftn(sa2D.*ch_fft_dict{1})));
                                %   a = repmat(imopen(ph2D>0.4272,strel('disk',20)),1,1,1);
                                %   a(:,end-80:end)=0;
                                %   ss_search = ss_search.*single(a);
                                %   subplot(2,8,cnt); imshow(ph.tiff(:,:,sidx), []); title(num2str(sidx));
                                %   subplot(2,8,cnt+8); imshow(ss_search, []); title(num2str(sidx)); cnt=cnt+1;
                                % end
                            end
                            
                            
                            %save(['Results/CHO/2D/' sig_type sn_type dir_list(n).name '.mat'], 'ss');
                            
                            
                            
                            resp_list_3d_search = [resp_list_3d_search; max(ss_3d_search(:))];
                            val_list = [val_list; max(crop(:))];
                            %disp([sig_type ' idx: ' num2str(n) ' Max_val: ' num2str(max(ss(:)))]);
                            
                        end
                        %disp('Added to the list');
                        %catch
                    end
                end
                %textprogressbar(100*n/numel(dir_list));
                
            end
            
            save(['Results/results_' flag_sigsize '_' flag_modelname '_' num2str(mtype_idx) '_' sn_type(1:end-1)],'resp_list_LKE', 'resp_list_search');
            save(['Results/results_3D_' flag_modelname '_' num2str(mtype_idx) '_' sn_type(1:end-1)],'resp_list_3d','resp_list_3d_search');
            %save(['Test_data/Crops_' sig_name '_' sn_name '.mat'], 'all_crops');
            %save(['Test_data/3D_Crops_' sig_name '_' sn_name '.mat'], 'all_crops_3d');
        end
        
    end
    
    
    for mtype_idx = 0:1
        if (mtype_idx==0)
            sig_type = 'CALC/';
        else
            sig_type = 'MASS/';
        end
        disp(sig_type);
        % calc performance
        load(['Results/results_2D_' flag_modelname '_' num2str(mtype_idx) '_Noise.mat']);
        neg_N = resp_list_LKE;
        neg_N_Search = resp_list_search;
        load(['Results/results_2D_' flag_modelname '_' num2str(mtype_idx) '_Signal.mat']);
        pos_S = resp_list_LKE;
        pos_S_Search = resp_list_search;
        [~,~,~,AUC,~] = perfcurve([zeros(length(neg_N),1); ones(length(pos_S),1)],[neg_N; pos_S],1);
        disp(['2D LKE:  '  num2str(AUC)]);
        
        [~,~,~,AUC,~] = perfcurve([zeros(length(neg_N_Search),1); ones(length(pos_S_Search),1)],[neg_N_Search; pos_S_Search],1);
        disp(['2D Search:  '  num2str(AUC)]);
        
        load(['Results/results_3D_' flag_modelname '_' num2str(mtype_idx) '_Noise.mat']);
        neg_N = resp_list_3d;
        neg_N_search = resp_list_3d_search;
        load(['Results/results_3D_' flag_modelname '_' num2str(mtype_idx) '_Signal.mat']);
        pos_S = resp_list_3d;
        pos_S_search = resp_list_3d_search;
        [~,~,~,AUC,~] = perfcurve([zeros(length(neg_N),1); ones(length(pos_S),1)],[neg_N; pos_S],1);
        disp(['3D LKE: '  num2str(AUC)]);
        
        [~,~,~,AUC,~] = perfcurve([zeros(length(neg_N_search),1); ones(length(pos_S_search),1)],[neg_N_search; pos_S_search],1);
        disp(['3D Search:  '  num2str(AUC)]);
    end
end