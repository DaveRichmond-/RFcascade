%% scripted version of LK algorithm with appearance variation and global shape normalizing transform

%% Run Pre-compute steps

clear all

% user defined dof in model

num_p = 1;
num_lambda = 1;

fname_model = 'som10-15_model.mat';
fname_image = 'Gblur_r=2.tif';
last_frame = 53;

[shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,SFP_positions,pixelList,SD,H,Trev_imageStack] = LK_precompute(fname_model,fname_image,last_frame,num_p,num_lambda);

%% initiate warp

%
load('fullDataSet.mat');

% select somite to track, and last frame (to start tracking from)
som_nums = [7];

for i=1:length(som_nums),
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    frame_indx = find(dataSet(som_indx,4) == last_frame);
    s_pos = dataSet(som_indx(frame_indx), 2:3)';
    s_pos_vec = reshape(permute(s_pos,[2 1]),[prod(size(s_pos)),1]);
    
    for j = 1:size(Sja,3),
        q_init(j,i) = dot(Sja_vec(:,j), (s_pos_vec - s0_vec));
    end

    [A_temp,b_temp] = LK_qtoA(q_init(:,i), weights);
    
    s_unq = (A_temp^-1)*(s_pos - repmat(b_temp,[1,size(s0,2)]));
    s_unq_vec = reshape(permute(s_unq,[2 1]),[prod(size(s_unq)) 1]);
    
    % back-calculate pp's
    
    for j = 1:size(Sj,3),
        
        p_init(j,i) = dot(Sj_vec(:,j), (s_unq_vec - s0_vec));
        
    end
    
    clear som_indx frame_inx s_pos A_temp b_temp s_unq
    
end

%% load all above steps

%clear
%load('precomputed.mat')
%load('precomputed_4pmodes.mat')
%load('reset_QPinit.mat')            % taken from best fit to first frame with no priors
%load('prior_model.mat')            % for using more sophisticated priors on the parameter updates

% reset initialization for frame22
%load('noPriors.mat','q_final_save','p_final_save')
%q_init = q_final_save(:,21);
%p_init = p_final_save(:,21);

% user set parameters

% priors on parameter smoothness
reg_weights = 1e-3*[1; 1; 1e-1; 1e-1; 1e3*ones(length(p_init),1)];

% parameters for positional priors
reg_weights_pos = 1e-3;                       % weight for positional prior

pos_nums = [7,8,1,3,4,5];

% general params on grad descent
step_size = 0.1*ones(4+length(p_init),1);
num_iters = 1000;
conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005*ones(length(p_init),1)];

%
load('som6_data.mat'),
load('som8_data.mat'),

%jump_thresh = 10;

%
conv_flag_save = NaN(2,size(Trev_imageStack,3));

%
for t = 1:size(Trev_imageStack,3),
    
    t,
    
    % extract position of somite 8 [p1,p8,p7]
    frame_num = last_frame - t + 1;                 % b/c movie starts at frame 58 and tracks backward
    s6_const = som6_data(find(som6_data(:,4)==frame_num),2:3)';
    s8_const = som8_data(find(som8_data(:,4)==frame_num),2:3)';
    
    %s_const = s8_const(:,[1,8,7]);
    %s_const = [s6_const(:,[5,4,3])];
    s_const = [s6_const(:,[5,4,3]) s8_const(:,[1,8,7])];
    
    % step frame #
    raw_image = Trev_imageStack(:,:,t);
    
    if t~=1,
        q_init = q_final_save(:,t-1);
        p_init = p_final_save(:,t-1); 
    end
    
    [q_final_save(:,t), p_final_save(:,t), sp_final_save(:,:,t), q, p, sp_save, conv_flag_save(1,t)] = LK_GradDescent_wPosPriors(raw_image,A0_vec,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,SFP_positions,pixelList,...
        H,SD,q_init,p_init,conv_thresh,num_iters,reg_weights,step_size,pos_nums,s_const,reg_weights_pos);
    
    if conv_flag_save(1,t) ~= 0,
        display(strcat('failed at time',num2str(t))),
        break
    end
    
    %{
    if conv_flag_save(1,t) ~= 0,
        
        %display('parameters jumped => increase strength of priors'),
        
        if t~=1,
            [q_init p_init] = update_priors(t, q_init_save, p_init_save, conv_flag_save, prior_model, 5);
        end
        
        [q_final_save(:,t), p_final_save(:,t), sp_final_save(:,:,t), q, p, sp_save, conv_flag_save(2,t)] = LK_GradDescent_wPriors(raw_image,A0_vec,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,SFP_positions,pixelList,...
            H,SD,q_init,p_init,conv_thresh,num_iters,10*reg_weights,0.1*step_size);
        
        if (conv_flag_save(2,t) ~= 0),
            error('still fails to converge, even with Priors'),
        end
        
    end
    %}
        
    p_init_save(:,t) = p_init;
    q_init_save(:,t) = q_init;
    
end

%% movie of convergence in a single frame

%t = 22;

%
save_flag = 0;
fname = strcat('frame # =',num2str(frame_num),'_POS=781345_c=0_c_pos=1e-2_ss=0p1_2pmodes.avi');
conv_time = size(p,2);

% plot somite 8 as well
frame_num = last_frame - t + 1;                 % b/c movie starts at frame 58 and tracks backward
s6_const = som6_data(find(som6_data(:,4)==frame_num),2:3)';
s8_const = som8_data(find(som8_data(:,4)==frame_num),2:3)';
s_const = [s6_const(:,[5,4,3])];

if save_flag,
    writerObj = VideoWriter(fname);
    open(writerObj);
end

set(0,'DefaultFigureWindowStyle','normal'),
fhandle = figure;

for iter = 1:3:conv_time,
    %iter,
    
    clf,
    subplot(2,3,1),
    imagesc(Trev_imageStack(:,:,t)),
    hold on,

    colormap('gray'),
    axis([231 392 150 550]),
    set(gca,'XTick',[],'YTick',[]),

    plot(sp_save(1,:,iter),sp_save(2,:,iter),'wo')
    plot(s_const(1,:),s_const(2,:),'ro')
    

    subplot(2,3,2)
    plot(q(3,1:iter)),
    
    subplot(2,3,3)
    plot(q(4,1:iter)),
    
    subplot(2,3,4)
    plot(q(1,1:iter)),
    
    subplot(2,3,5)
    plot(q(2,1:iter)),
    
    subplot(2,3,6)
    plot(p(1,1:iter)),
    
    % 
    
    %
    if save_flag,
        frame = getframe(fhandle);
        writeVideo(writerObj,frame);
    else
        pause(0.25),
    end
end

if save_flag,
    close(writerObj),
end

set(0,'DefaultFigureWindowStyle','docked'),

%% movie of full tracking

save_flag = 1;
fname = 'POS781345_SS=0.1_c=1e-3*ones_c_pos=1e-3_ManualTrackingPriors.avi';

if save_flag,    
    writerObj = VideoWriter(fname);
    open(writerObj);
end

set(0,'DefaultFigureWindowStyle','normal'),
fhandle = figure;

for t = 1:size(sp_final_save,3),
    
    imagesc(Trev_imageStack(:,:,t)),
    colormap('gray'),
    axis([231 392 150 550]),
    set(gca,'XTick',[],'YTick',[]),

    hold on,
    
    % extract manual landmark positions from dataSet
    %{
    som_indx = find(dataSet(:,7) == som_num);
    frame_indx = find(dataSet(som_indx,4) == i);
    manual_pos = dataSet(som_indx(frame_indx), 2:3)';
    %}
    
    %plot(manual_pos(1,:),manual_pos(2,:),'wo');
    
    % 
    frame_num = last_frame - t + 1;                 % b/c movie starts at frame 58 and tracks backward
    
    s6_const = som6_data(find(som6_data(:,4)==frame_num),2:3)';
    s8_const = som8_data(find(som8_data(:,4)==frame_num),2:3)';
    %s_const = s8_const(:,[1,8,7]);
    %s_const = [s6_const(:,[5,4,3])];
    s_const = [s6_const(:,[5,4,3]) s8_const(:,[1,8,7])];
    
    plot(sp_final_save(1,:,t),sp_final_save(2,:,t),'wo')
    plot(s_const(1,:),s_const(2,:),'ro')
    
    if 1-save_flag,
        pause(0.25),
    end
    
    if save_flag,
        frame = getframe(fhandle);
        writeVideo(writerObj,frame);
    end
end

if save_flag,
    close(writerObj),
end

set(0,'DefaultFigureWindowStyle','docked'),

%% plot and save parameters

prefix = 'tracking_c=0_';
%folder = 'hierGradDesc_2rounds_c=ones';

%cd('frame58')
%mkdir(folder),
%cd(folder),

set(0,'DefaultFigureWindowStyle','docked'),

for i = 1:size(q,1),
    figure,
    plot(q(i,:)),
    print('-dtiff',strcat(prefix,'q',num2str(i),'.tif'))
    
end
    
for i = 1:size(p,1),
    figure,
    plot(p(i,:)),
    print('-dtiff',strcat(prefix,'p',num2str(i),'.tif'))
    
end

%cd ../..

%%
