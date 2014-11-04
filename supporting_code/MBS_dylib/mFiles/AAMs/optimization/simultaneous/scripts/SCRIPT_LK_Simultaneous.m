%% scripted version of LK algorithm with appearance variation and global shape normalizing transform

%% Pre-compute steps

clear all

%% user defined dof in model

num_p = 1;
num_lambda = 1;

fname_model = 'som7_model.mat';
fname_image = 'f1-58_blur.tif';
last_frame = 58;

[shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,Ai,Ai_vec,Grad_A0_vec,Grad_Ai_vec,dWdp,dNdq,SFP_positions,pixelList,Trev_imageStack] = LK_precompute_Simultaneous(fname_model,fname_image,last_frame,num_p,num_lambda);

%% initiate warp

%
load('fullDataSet.mat'),

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

L_init = zeros(num_lambda,1);
L_init(1) = 0.2;

%% Iterate steps

% user set parameters
reg_weights = 1e-3*[1; 1; 1e-1; 1e-1; 1e3; 0];
step_size = ones(6,1);
num_iters = 500;
conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005; 0.0002];

%
conv_flag_save = NaN(1,size(Trev_imageStack,3));

%
figure,
for t = 1:size(Trev_imageStack,3),
    
    t,
    
    % step frame #
    raw_image = Trev_imageStack(:,:,t);
    
    if t~=1,
        q_init = q_final_save(:,t-1);
        p_init = p_final_save(:,t-1); 
    end
    
    [q_final_save(:,t), p_final_save(:,t), L_final(:,t), sp_final_save(:,:,t), q, p, L, sp_save, conv_flag_save(t)] = LK_GradDescent_Sim_wPriors(raw_image,A0_vec,Grad_A0_vec,Ai_vec,Grad_Ai_vec,dNdq,dWdp,...
        s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,SFP_positions,pixelList,q_init,p_init,L_init,conv_thresh,num_iters,reg_weights,step_size);
    
    if conv_flag_save(1,t) ~= 0,
        break,
    end
    
    p_init_save(:,t) = p_init;
    q_init_save(:,t) = q_init;
    
    % display
    
    imagesc(Trev_imageStack(:,:,t)),
    colormap('gray'),
    axis([231 392 150 550]),
    %axis([256 768 512 1024]),
    set(gca,'XTick',[],'YTick',[]),

    hold on,
    
    plot(sp_final_save(1,:,t),sp_final_save(2,:,t),'wo')
    pause(1)
     
end

%% movie of convergence in a single frame

%t = 22;

save_flag = 0;
fname = 'temp.avi';%strcat('t=',num2str(t),'_c=1e-3*ones_ss=0p3.avi');
conv_time = size(p,2);

if save_flag,
    writerObj = VideoWriter(fname);
    open(writerObj);
end

set(0,'DefaultFigureWindowStyle','normal'),
fhandle = figure;

for iter = 1:conv_time,
    %iter,
    
    clf,
    subplot(2,4,1),
    imagesc(Trev_imageStack(:,:,t)),
    hold on,

    colormap('gray'),
    axis([231 392 150 550]),
    %axis([256 768 512 1024]),
    set(gca,'XTick',[],'YTick',[]),

    plot(sp_save(1,:,iter),sp_save(2,:,iter),'wo')

    subplot(2,4,2)
    plot(q(3,1:iter)),
    
    subplot(2,4,3)
    plot(q(4,1:iter)),
    
    subplot(2,4,4)
    plot(q(1,1:iter)),
    
    subplot(2,4,5)
    plot(q(2,1:iter)),
    
    subplot(2,4,6)
    plot(p(1,1:iter)),
    
    subplot(2,4,7)
    plot(L(1,1:iter)),
    
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
fname = 'priors=1e-3*[1; 1; 1e-1; 1e-1; 1e3; 1e4]_SIM.avi';

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
    
    plot(sp_final_save(1,:,t),sp_final_save(2,:,t),'wo')
    
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
