%% scripted version of LK algorithm with appearance variation and global shape normalizing transform

%% Pre-compute steps

clear all

%% user defined dof in model

num_p = 4;
num_lambda = 50;

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

% (*) redefine SFP (necessary due to shrinkage at borders from gradient)

% calculate indices into pixelList where elements are non-NaN, and then apply to pixelList

SFP_indx = find(~isnan(dA0dx(:)) .* ~isnan(dA0dy(:)));

% apply this domain to all 2D images to create condensed vector form

A0_vec = A0(SFP_indx);

for i = 1:size(Ai,3),
    
    temp = Ai(:,:,i);
    Ai_vec(:,i) = temp(SFP_indx);
    
end
clear temp

Grad_A0_vec(:,1) = dA0dx(SFP_indx);
Grad_A0_vec(:,2) = dA0dy(SFP_indx);
SFP_positions = [X(SFP_indx) Y(SFP_indx)]';

% (*) REDUCE DIMENSIONALITY OF MODEL FOR SPEED

Sj = Sj(:,:,1:num_p);
Sj_vec = Sj_vec(:,1:num_p);
Ai_vec = Ai_vec(:,1:num_lambda);

% (4) evaluate the Jacobians

[dWdp, dNdq] = LK_warpJacobian(X, Y, SFP_indx, s0, Sj, Sja);

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
s_init = s0_vec + shape.Psi(:,1:num_p)*p;

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

% plot
%{
figure, hold on,
plot([340 349 346 313 283 272 280 306], [436 453 471 463 497 478 450 423],'ko')
plot(init_pos(1,:),init_pos(2,:),'ro')
%}

q = NaN(4,1);
q(1) = (T(1,1) - 1) * weights(1);
q(2) = T(2,1) * weights(2);
q(3) = s_pos_mean(1) * weights(3);
q(4) = s_pos_mean(2) * weights(4);

q_old = q;
p_old = p;

%% Iterate steps

% user parameters

thresh = 1e-6;

% initialize

num_iters = NaN(1,size(Trev_imageStack,3));

% perturb warp initialization
%{
q_old = q + (rand(size(q,1),1)*20*weights(3) - 10);
p_old = p + (rand(size(p,1),1)*0.1 - 0.05);
%}

% calculate thresholds for fitting parameters
%{
thresh_p = diag(sqrt(shape.Lambda(1:num_p,1:num_p)));
thresh_q = [ 5*weights(3) 5*weights(4)];
%}

tic,

for i = 1:size(Trev_imageStack,3),
    
    i,
    k = 0;
    
    % set raw_image to current frame
    
    raw_image = Trev_imageStack(:,:,i);
    w_image_vec = NaN(size(A0_vec));
    
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
    
    % clear unneeded params
    
    clear st1 s1 raw_image w_positions w_image_vec Err_image_vec SD_update Delta_qp dq dp q_new p_new
    
end

% reverse tracking

fit_position = flipdim(fit_position,3);

toc,

%}

%% display results

fname = strcat('LK tracking of somite 7 with ',num2str(num_p),' shape parameters.avi');
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


% look at difference images (eg. Error image)

[im] = make_image_from_image_vector(Err_image_vec, X, SFP_indx);

figure,
imagesc(im),
colormap('gray')
%}

%% Post-compute steps

