%% scripted version of LK algorithm with appearance variation and global shape normalizing transform

%% Run Pre-compute steps

clear all

% user defined dof in model

num_p = 1;
num_lambda = 1;

fname_model = 'som10-15_7ptModel_3.mat';
fname_image = 'Gblur_r=2.tif';
last_frame = 111;

[shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,SFP_positions,pixelList,SD,H,Trev_imageStack] = LK_precompute(fname_model,fname_image,last_frame,num_p,num_lambda);

%% initiate warp

%load('fullDataSet.mat');
load('som10-22_frame111_init.mat'),

% select somite to track, and last frame (to start tracking from)
som_nums = [10];

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
%load('reset_QPinit.mat')
%load('prior_model.mat')

% reset initialization for frame22
%{
load('noPriors.mat','q_init_save','p_init_save')
q_init = q_init_save;
p_init = p_init_save;
%}

% Iterate steps

% user set parameters
reg_weights = 1e-3*[1; 1; 1e-1; 1e-1; 1e3];
step_size = 0.1*ones(5,1);
num_iters = 1000;
conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005];
jump_thresh = 10;

%
conv_flag_save = NaN(2,size(Trev_imageStack,3));

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
    
    [q_final_save(:,t), p_final_save(:,t), sp_final_save(:,:,t), q, p, sp_save, conv_flag_save(1,t)] = LK_GradDescent_wPriors(raw_image,A0_vec,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,SFP_positions,pixelList,...
        H,SD,q_init,p_init,conv_thresh,num_iters,reg_weights,step_size);
    
    if conv_flag_save(1,t) ~= 0,
        break,
    end
    
    % test if parameters jump significantly between frames
    %{
    if t > 5,
        diff_q(:,t) = q_final_save(:,t) - q_final_save(:,t-1);
        diff_p(:,t) = p_final_save(:,t) - p_final_save(:,t-1);
        
        test_jump = norm([diff_q(:,t); diff_p(:,t)]);
    else
        test_jump = 0;
    end
    %}  
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
    
    % save q,p
    %{
    q_save(:,t) = q_old;
    p_save(:,t) = p_old;
    sp_save(:,:,t) = sp;
    cost_save(:,t) = cost(i);
    %}
    
    % display
    imagesc(Trev_imageStack(:,:,t)),
    colormap('gray'),
    %axis([231 392 150 550]),
    axis([256 768 512 1024]),
    set(gca,'XTick',[],'YTick',[]),

    hold on,
    
    plot(sp_final_save(1,:,t),sp_final_save(2,:,t),'wo')
    pause(1)
    
end

%{
% <NEW>
% user set parameters
q_weights = [0; 0; 0; 0];
p_weights = zeros(length(p_init),1);
reg_weight = [q_weights; p_weights];  % cost, i.e. scaling of regularization term
% </NEW>

clear cost sp_save store_dp store_dq

% initialize
thresh = eps;
w_image_vec = NaN(size(A0_vec));
q_old = q_init;
p_old = p_init;
q0 = q_init;
p0 = p_init;
c = sqrt(reg_weight);   % simpler to use the sqrt(reg_weight) for the math... 

for t = 1:size(Trev_imageStack,3),
    
    t,
    
    % step frame #
    raw_image = Trev_imageStack(:,:,t);

    % grad descent loop
    for i = 1:100,
        
        % (1) Warp I with W(x;p) followed by N(x;q); where p,q come from last iteration ---------------->
        
        % calculate TSP warp from movement of base-mesh points under p_old, q_old
        [st1, s1] = LK_Warp(s0,Sj,Sja,weights,q_old,p_old);
        
        % store shape during grad descent
        %{
        if i == 1,
            sp_save(:,:,i) = s1;
        end
        %}
        
        w_positions = fnval(st1, SFP_positions);
        
        for m = 1:length(pixelList),
            w_image_vec(m) = bilinear_interp(raw_image, w_positions(2,m), w_positions(1,m));
        end
        
        % normalize w_image_vec with respect to first guess of position
        if i==1,
            w_image_vec_MEAN = repmat(mean(w_image_vec), [size(w_image_vec,1),1]);
            w_image_vec_STD = repmat(sqrt(sum(w_image_vec.^2,1)), [size(w_image_vec,1),1]);
        end
        w_image_vec = w_image_vec - w_image_vec_MEAN;
        w_image_vec = w_image_vec ./ w_image_vec_STD;
        
        % (2) Compute the error image
        Err_image_vec = w_image_vec - A0_vec;
        
        % <NEW>
        
        d = [q_old; p_old] - [q0; p0];
        
        % (2 cont'd) Compute F(p)
        F = sqrt((c.*c)'*(d.*d));
        
        % (5a) Compute dF/dP and dp'/dDp
        if F == 0,
            dFdp = zeros(size(d));        % c.*sign(d);                                      % JUST RETURNS ZERO
        else
            dFdp = ((c.*c).*d) / F;
        end
        
        % calc reparameterization
        dppdDp = diag(-[1 1 1 1 1]);                                 % LINEAR APPROX FOR NOW
        %{
        if i==1,
            store_p_old = p_old';
            store_q_old = q_old';
            store_dq = zeros(size(q_init))';
            store_dp = zeros(size(p_init))';
        end
        dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,store_q_old,store_p_old,store_dq,store_dp,q_old,p_old);
        %}
        
        % (6a) calc Hessian and invert
        H_prior = (dFdp'*dppdDp)'*(dFdp'*dppdDp);
        inv_H = (H + H_prior)^-1;
        
        % (7) Compute SD update
        SD_update = SD' * Err_image_vec;
        SD_update = SD_update - (dFdp'*dppdDp)'*F;
        
        % </NEW>        
        
        % (8) Compute the parameter updates: (dq,dp)
        Delta_qp = inv_H * SD_update;                                   % MINUS SIGN MISSING ???
        dq = Delta_qp(1:4);
        dp = Delta_qp(5:end);
        
        % store updates for troubleshooting
        %{
    store_dq(:,i) = dq(q_indx)';
    store_dp(i) = dp(p_indx)';
        %}
        
        % (9) Update parameters by (dq,dp)
        [q_new, p_new, sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q_old,p_old,dq,dp);
        
        % store for troubleshooting
        store_q_old = q_old;
        store_p_old = p_old;
        store_dq = dq;
        store_dp = dp;
        %{
    diff_q = q_new' - q_old;
    diff_p = p_new' - p_old;
    
    store_diff_qp(:,i) = [diff_q; diff_p];
    
    
    % test convergence
    diff_q = abs(q_new' - q_old);
    diff_p = abs(p_new' - p_old);
        %}
        
        % re-assign for next iteration
        q_old = q_new';
        p_old = p_new';
        
        % store position
        %sp_save(:,:,i+1) = sp;
        cost(i) = sum(Err_image_vec.^2);
        
        if (i ~= 1) && ((cost(i-1) - cost(i)) < thresh),
            break
        end
        
    end
    
    % save q,p
    q_save(:,t) = q_old;
    p_save(:,t) = p_old;
    sp_save(:,:,t) = sp;
    cost_save(:,t) = cost(i);
    
    % update prior for next time point
    q0 = q_save(:,t);
    p0 = p_save(:,t);
    
end
%}

%% save output for making movies

save(strcat('som#',num2str(som_nums),'_tracking_7ptModel_f111.mat'),'q_final_save','p_final_save','sp_final_save','conv_flag_save')

%% remake movie from saved data
%{
indx = find(~isnan(conv_flag_save(1,:)));
div_frames = find(conv_flag_save(1,indx) ~= 0);
conv_frames = [1:div_frames(1)-1];
%}

%{
%
save_flag = 0;
fname = strcat('som#',num2str(som_nums),'_SS=0.1_c=1e-3_f111.avi');

if save_flag,
    
    writerObj = VideoWriter(fname);
    open(writerObj);

end

set(0,'DefaultFigureWindowStyle','normal'),
fhandle = figure;

for t = 1:size(sp_final_save,3),
    
    imagesc(Trev_imageStack(:,:,t)),
    colormap('gray'),
    %axis([231 392 150 550]),
    axis([256 768 512 1024]),
    set(gca,'XTick',[],'YTick',[]),

    hold on,
    
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
%}

%% movie of full tracking

save_flag = 1;
fname = strcat('som#',num2str(som_nums),'_SS=0.1_c=1e-3_f111.avi');

if save_flag,
    
    writerObj = VideoWriter(fname);
    open(writerObj);

end

set(0,'DefaultFigureWindowStyle','normal'),
fhandle = figure;

for t = 1:size(sp_final_save,3),
    
    imagesc(Trev_imageStack(:,:,t)),
    colormap('gray'),
    %axis([231 392 150 550]),
    axis([256 768 512 1024]),
    set(gca,'XTick',[],'YTick',[]),

    hold on,
    
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

%% movie of convergence in a single frame

%t = 22;

save_flag = 0;
fname = 'temp.avi';%strcat('t=',num2str(t),'_c=1e-3*ones_ss=0p3.avi');
conv_time = 100;%size(p,2);

if save_flag,
    writerObj = VideoWriter(fname);
    open(writerObj);
end

set(0,'DefaultFigureWindowStyle','normal'),
fhandle = figure;

for iter = 1:conv_time,
    iter,
    
    clf,
    subplot(2,3,1),
    imagesc(Trev_imageStack(:,:,t)),
    hold on,

    colormap('gray'),
    %axis([231 392 150 550]),
    axis([256 768 512 1024]),
    set(gca,'XTick',[],'YTick',[]),

    plot(sp_save(1,:,iter),sp_save(2,:,iter),'wo')

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

figure,
imagesc(A0),
colormap('gray')

for i = 1:9,
    figure(i+1),
    imagesc(Ai(:,:,i))
    colormap('gray')
end













%% display first and last shapes for test of convergence

figure
imagesc(raw_image),
colormap('gray'),

hold on

plot(sp_save(1,:,1),sp_save(2,:,1),'ko')
plot(sp_save(1,:,end),sp_save(2,:,end),'wo')
s_pos = dataSet(som_indx(frame_indx), 2:3)';
plot(s_pos(1,:),s_pos(2,:),'ro')

axis([231 392 400 550]),

% figure of corrections
%{
figure,
plot(store_dq),
title('parameter')
%}

% cost vs. iteration #

figure,
plot(cost)
title('cost')

%{

% user parameters

thresh = 1e-6;

% initialize

w_image_vec = NaN(size(A0_vec));
num_iters = NaN(1,size(Trev_imageStack,3));

q_old = q; % + (rand(size(q,1),1)*20*weights(3) - 10);
p_old = p; % + (rand(size(p,1),1)*0.1 - 0.05);

% calculate thresholds for fitting parameters
%{
thresh_p = diag(sqrt(shape.Lambda(1:num_p,1:num_p)));
thresh_q = [ 5*weights(3) 5*weights(4)];
%}

tic,

for i = 1:1, %1:size(Trev_imageStack,3),
    
    i,
    k = 0;
    
    % set raw_image to current frame
    
    raw_image = Trev_imageStack(:,:,i);
    
    % reset diff_q,diff_p above threshold
    
    diff_p = 10*thresh*ones(num_p,1);
    diff_q = 10*thresh*ones(4,1);

    
    while (sum([diff_q; diff_p] > thresh*ones(4+num_p,1)) ~= 0),
        
        k = k+1;
        
        % (1) Warp I with W(x;p) followed by N(x;q); where p,q come from last iteration ---------------->
        
        % calculate TSP warp from movement of base-mesh points under p,q
        
        [st1, s1] = LK_Warp(s0,Sj,Sja,weights,q_old,p_old);
        
        % store shape during grad descent
        %{
    if i == 1,
        
        sp_save(:,:,i) = s1;
        
    end
        %}
        
        w_positions = fnval(st1, SFP_positions);
        
        for m = 1:length(SFP_indx),
            
            w_image_vec(m) = bilinear_interp(raw_image, w_positions(2,m), w_positions(1,m));
            
        end
        
        % (2) Compute the error image
        
        Err_image_vec = w_image_vec - A0_vec;
        
        % (7) Compute SD update
        
        SD_update = SD' * Err_image_vec;
        
        % (8) Compute the parameter updates: (dq,dp)
        
        Delta_qp = inv_H * SD_update;
        dq = Delta_qp(1:4);
        dp = Delta_qp(5:end);
        
        % (9) Update parameters by (dq,dp)
        
        [q_new, p_new, sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q,p,dq,dp);
        
        % convergence?
        
        diff_q = abs(q_new' - q_old);
        diff_p = abs(p_new' - p_old);
        
        store(i).diff_qp(:,k) = [diff_q; diff_p];
        
        % re-assign for next iteration
        
        q_old = q_new';
        p_old = p_new';
        
    end

    % store results from ith frame
    
    [st_final, s_final] = LK_Warp(s0,Sj,Sja,weights,q_new,p_new);
    
    fit_position(:,:,i) = s_final;
    num_iters(i) = k;
    
end

% reverse tracking

fit_position = flipdim(fit_position,3);

toc,

%}

%% look at difference images (eg. Error image)

[w_im] = make_image_from_image_vector(w_image_vec, X, pixelList);
[Err_im] = make_image_from_image_vector(Err_image_vec, X, pixelList);

figure,
imagesc(A0)
colormap('gray')

figure,
imagesc(w_im),
colormap('gray')

figure,
imagesc(Err_im),
colormap('gray')

%% save

%Err_im = scaleImage(Err_im, '16bit', 1);
Err_im = Err_im - min(Err_im(:));
bfsave(Err_im,'Err_im_TEST.tif')

A0_im = scaleImage(A0, '16bit', 1);
bfsave(A0_im,'A0_im_TEST.tif')

w_im = scaleImage(w_im, '16bit', 1);
bfsave(w_im,'w_im_TEST.tif')

%%

% test convergence

figure, hold on,
for i = 1:size(diff_q,1)
    
    plot(store_diff_qp(i,:),'k')

end

figure, hold on,
for i = 1:size(diff_p,1)
    
    plot(store_diff_qp(i+4,:),'k')

end

%% display results

fname = strcat('LK fitting somite 7 with ',num2str(num_p),' shape parameters.avi');
writerObj = VideoWriter(fname);
open(writerObj);

fhandle = figure;

for i = 1:size(imageStack,3);
    
    imagesc(imageStack(:,:,i)),
    colormap('gray'),
    axis([231 392 275 550]),
    set(gca,'XTick',[],'YTick',[]),

    hold on,
    
    % extract manual landmark positions from dataSet
    
    som_indx = find(dataSet(:,7) == som_num);
    frame_indx = find(dataSet(som_indx,4) == i);
    manual_pos = dataSet(som_indx(frame_indx), 2:3)';

    
    plot(manual_pos(1,:),manual_pos(2,:),'wo');
    
    plot(fit_position(1,:,i),fit_position(2,:,i),'ro')
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
    
end

close(writerObj),

% test convergence
%{ 
figure, hold on,
for i = 1:size(diff_q,1)
    
    plot(diff_q(i,:),'k')

end

figure, hold on,
for i = 1:size(diff_p,1)
    
    plot(diff_p(i,:),'k')

end

% movie of convergence

figure
imagesc(raw_image),
colormap('gray'),

hold on

plot(sp_save(1,:,1),sp_save(2,:,1),'bo')
plot(sp_save(1,:,end),sp_save(2,:,end),'ro')


%
%}

%% movie of convergence

%% display results

fname = strcat('np=',num2str(num_p),'_nlambda=',num2str(num_lambda),'.avi');
writerObj = VideoWriter(fname);
open(writerObj);

fhandle = figure;

for i = 1:size(sp_save,3);
    
    imagesc(Trev_imageStack(:,:,1)),
    colormap('gray'),
    axis([231 392 350 550]),
    set(gca,'XTick',[],'YTick',[]),

    hold on,
    
    % extract manual landmark positions from dataSet
    %{
    som_indx = find(dataSet(:,7) == som_num);
    frame_indx = find(dataSet(som_indx,4) == i);
    manual_pos = dataSet(som_indx(frame_indx), 2:3)';
    %}
    
    %plot(manual_pos(1,:),manual_pos(2,:),'wo');
    
    plot(sp_save(1,:,i),sp_save(2,:,i),'wo')
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
    
end

close(writerObj),

%% Post-compute steps

