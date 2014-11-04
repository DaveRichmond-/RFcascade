%% Pre-compute steps

clear all

%% user defined dof in model

num_p = 1;
num_lambda = 1;

fname_model = 'som7_model.mat';
fname_image = 'Focused 120807_bf_f0000_frame103_Gblur_r=2.tif';
last_frame = 1;

[shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,Ai,Ai_vec,SFP_positions,pixelList,SD,H,Trev_imageStack] = ...
    LK_precompute(fname_model,fname_image,last_frame,num_p,num_lambda);

%% initiate warp

%
load('f0000_boundary_labels.mat'),

% select somite boundary to initialize
som_boundaries = [10];

% initialize shape
p_init = -0.3;

for i=1:length(som_boundaries),
    
    % read in boundary positions from table
    boundary_indx = find(dataSet(:,1) == som_boundaries(i));
    s8_annot = dataSet(boundary_indx, 2:3)';
    s4_annot = dataSet(boundary_indx+1, 2:3)';
    
    %
    s_annot = [s8_annot, s4_annot];
    s_model = s0 + p_init*Sj;
    s_model = s_model(:,[8,4]);
    
    % align
    [x_aligned,A,b] = align_model_w_annotation(s_model,s_annot);
    
    % pull q_init's from A,b
    q_init(1,i) = (A(1,1) - 1) * weights(1);
    q_init(2,i) = A(2,1) * weights(2);
    q_init(3,i) = b(1) * weights(3);
    q_init(4,i) = b(2) * weights(4);
    
    clear boundary_indx
    
    %{
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
    %}
    
end

%%

% parameters for positional priors
reg_weights_pos = 1e-10;
reg_weights = zeros(5,1);

pos_nums = [8,4];

% general params on grad descent
step_size = 0.2*ones(4+length(p_init),1);
num_iters = 500;
conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005];

%
raw_image = Trev_imageStack;

for s = 1:length(som_boundaries),
    
    s,
    
    % extract position of somite [p8,p4]
    s_const = s_annot;
    
    %{
    frame_num = last_frame - t + 1;                 % b/c movie starts at frame 58 and tracks backward
    s6_const = som6_data(find(som6_data(:,4)==frame_num),2:3)';
    s8_const = som8_data(find(som8_data(:,4)==frame_num),2:3)';
    
    %s_const = s8_const(:,[1,8,7]);
    %s_const = [s6_const(:,[5,4,3])];
    s_const = [s6_const(:,[5,4,3]) s8_const(:,[1,8,7])];
    %}
    
    [q_final_save(:,s), p_final_save(:,s), sp_final_save(:,:,s), q, p, sp_save, conv_flag_save(s)] = ...
        LK_GradDescent_wPosPriors(raw_image,A0_vec,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,SFP_positions,pixelList,...
        H,SD,q_init,p_init,conv_thresh,num_iters,reg_weights,step_size,pos_nums,s_const,reg_weights_pos);
    
end

%%

set(0,'DefaultFigureWindowStyle','docked'),
%
save_flag = 0;
fname = 'somite10_frame103_fit.avi';
conv_time = size(p,2);

if save_flag,
    writerObj = VideoWriter(fname);
    open(writerObj);
    set(0,'DefaultFigureWindowStyle','normal'),
end

fhandle = figure;

for iter = 1:5:conv_time,
    %iter,
    
    clf,
    subplot(2,3,1),
    imagesc(raw_image),
    hold on,

    colormap('gray'),
    axis([0.2 0.4 0.5 0.75]*1024)
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
    if save_flag,
        frame = getframe(fhandle);
        writeVideo(writerObj,frame);
    else
        pause(0.5),
    end
end

if save_flag,
    close(writerObj),
    set(0,'DefaultFigureWindowStyle','docked'),
end