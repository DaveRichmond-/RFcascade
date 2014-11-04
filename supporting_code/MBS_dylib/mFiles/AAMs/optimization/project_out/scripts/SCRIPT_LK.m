%% scripted version of LK algorithm with appearance variation and global shape normalizing transform

%% Pre-compute steps

clear all

%% user defined dof in model

num_p = 1;
num_lambda = 1;

%% load data

load('som7_model.mat');

% basic properties of shape

shape_points = 8;
shape_dim = 2;
shape_dof = shape_points * shape_dim;

% define characteristic matrices from model

s0 = reshape(shape(1).xbar_vector, [shape_points, shape_dim])';
Sj = permute(reshape(shape(1).Psi, [shape_points, shape_dim, shape_dof]), [2 1 3]);

% vector representations                    IN FUTURE, WORK WITH VECTOR REPRESENTATION.  YOU RARELY NEED MATRIX REPRESENTATION!

s0_vec  = reshape(permute(s0, [2 1]), [shape_dof 1]);
Sj_vec  = squeeze(reshape(permute(Sj, [2 1 3]), [shape_dof 1 shape_dof]));

% make corresponding e-vectors of Similarity Transform

[Sja, Sja_vec, weights] = LK_SimTransfBasis(s0);

% basic properties of appearance

% 

appear_pixels = size(appear.Psi,1);
appear_dof = size(appear.Psi,2);

% construct template and eigenimages as 2D images

[A0, Ai] = mean_eigen_images(appear.gbar, X, pixelList, appear.Psi);

%% (3) Evaluate gradient of template ---------->

% calculate gradient

[dA0dx dA0dy] = gradient(A0, X(1,:), Y(:,1));

% calculate indices into pixelList where elements are non-NaN, and then apply to pixelList

Grad_indx = find(~isnan(dA0dx(:)) .* ~isnan(dA0dy(:)));
Edge_indx = pixelList(find(1 - ismember(pixelList,Grad_indx)));

% remove NaN's at edge of gradient image (replaced by zeros)

dA0dx(Edge_indx) = 0;
dA0dy(Edge_indx) = 0;

% apply this domain to all 2D images to create condensed vector form

A0_vec = A0(pixelList);

% replace next step with reshape()

for i = 1:size(Ai,3),
    
    temp = Ai(:,:,i);
    Ai_vec(:,i) = temp(pixelList);
    
end
clear temp

%

Grad_A0_vec(:,1) = dA0dx(pixelList);
Grad_A0_vec(:,2) = dA0dy(pixelList);
SFP_positions = [X(pixelList) Y(pixelList)]';

% (*) REDUCE DIMENSIONALITY OF MODEL FOR SPEED

Sj = Sj(:,:,1:num_p);
Sj_vec = Sj_vec(:,1:num_p);
Ai_vec = Ai_vec(:,1:num_lambda);

% (4) evaluate the Jacobians

[dWdp, dNdq] = LK_warpJacobian(X, Y, pixelList, s0, Sj, Sja);

% (5) compute modified steepest descent images

SD = LK_SDimage(Grad_A0_vec, dNdq, dWdp, Ai_vec);

% (6) compute Hessian and inverse Hessian

H = LK_Hessian(SD);
inv_H = H^-1;

%% load raw data

filename = 'f1-58_blur.tif';
[imageStack,sizeC,sizeZ,sizeT] = bf_openStack(filename);
imageStack = double(imageStack);
imageStack_vec = reshape(imageStack, [prod([size(imageStack,1),size(imageStack,2)]), size(imageStack,3)]);

% normalize the graylevel values in the image

% translate imageStack vectors to be zero mean
imageStack_vec = imageStack_vec - repmat(mean(imageStack_vec,1), [size(imageStack_vec,1),1]);

% make raw image vector one-variance !!!!!!!!!!!!!!!!!!!!
imageStack_vec = imageStack_vec ./ repmat(sqrt(sum(imageStack_vec.^2,1)), [size(imageStack_vec,1),1]);
imageStack = reshape(imageStack_vec, size(imageStack));

% invert imageStack time, to track backwards
Trev_imageStack = flipdim(imageStack,3);

% display
%{
figure,
imagesc(imageStack(:,:,1));
colormap gray
%}

%% initiate warp

%
load('fullDataSet.mat');

% select somite to track, and last frame (to start tracking from)
som_num = shape.som_num;
last_frame = shape.frames(end);

% initiate shape to best fit of model to known manually 
p = shape.b(1:num_p, last_frame);

% RE-WRITTEN USING Sj
s_init = s0_vec + Sj_vec(:,1:num_p)*p;

% solve for q
som_indx = find(dataSet(:,7) == som_num);
frame_indx = find(dataSet(som_indx,4) == last_frame);
s_pos = dataSet(som_indx(frame_indx), 2:3)';
s_pos_mean = mean(s_pos,2);
s_pos = s_pos - repmat(s_pos_mean,[1 8]);

s_pos_vec = s_pos';
s_pos_vec = s_pos_vec(:);

[dummy, T] = align_shape_vectors_wParams(s_init, s_pos_vec);

init_pos = T*reshape(s_init,[8 2])' + repmat(s_pos_mean, [1 8]);

q = NaN(4,1);
q(1) = (T(1,1) - 1) * weights(1);
q(2) = T(2,1) * weights(2);
q(3) = s_pos_mean(1) * weights(3);
q(4) = s_pos_mean(2) * weights(4);

%% Iterate steps

% initialize
w_image_vec = NaN(size(A0_vec));
raw_image = Trev_imageStack(:,:,1);         % step image forward (actually, backward) one frame

clear cost sp_save store_dp store_dq

[q_new,p_new,sp] = iterate_LK(s0,Sj,Sja,weights,SFP_positions,pixelList,raw_image,A0_vec,SD,inv_H,q,p,eps);

%%

%{ 

visualize

.
.
.
v


%}


















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

