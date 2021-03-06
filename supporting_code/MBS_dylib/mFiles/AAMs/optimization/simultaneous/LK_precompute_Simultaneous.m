function [shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,Ai,Ai_vec,Grad_A0_vec,Grad_Ai_vec,dWdp,dNdq,SFP_positions,pixelList,Trev_imageStack] = LK_precompute_Simultaneous(fname_model,fname_image,last_frame,num_p,num_lambda);

% load data

load(fname_model);

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

appear_pixels = size(appear.Psi,1);
appear_dof = size(appear.Psi,2);

% construct template and eigenimages as 2D images

[A0, Ai] = mean_eigen_images(appear.gbar, X, pixelList, appear.Psi);

% (3) Evaluate gradient of template, A0, and eigen images, Ai ---------->

% calculate gradient
[dA0dx dA0dy] = gradient(A0, X(1,:), Y(:,1));

% calculate indices into pixelList where elements are non-NaN, and then apply to pixelList
Grad_indx = find(~isnan(dA0dx(:)) .* ~isnan(dA0dy(:)));
Edge_indx = pixelList(find(1 - ismember(pixelList,Grad_indx)));

% remove NaN's at edge of gradient image (replaced by zeros)
dA0dx(Edge_indx) = 0;
dA0dy(Edge_indx) = 0;

%
Grad_A0_vec(:,1) = dA0dx(pixelList);
Grad_A0_vec(:,2) = dA0dy(pixelList);

% apply this domain to all 2D images to create condensed vector form
A0_vec = A0(pixelList);


% repeat for eigen images

Ai_vec = reshape(Ai,[prod([size(Ai,1),size(Ai,2)]), size(Ai,3)]);
Ai_vec = Ai_vec(pixelList,:);

for i = 1:size(Ai,3),
    [dAidx(:,:,i) dAidy(:,:,i)] = gradient(Ai(:,:,i), X(1,:), Y(:,1));
end

dAidx_vec = reshape(dAidx,[prod([size(dAidx,1),size(dAidx,2)]) size(dAidx,3)]);
dAidy_vec = reshape(dAidy,[prod([size(dAidy,1),size(dAidy,2)]) size(dAidy,3)]);

dAidx_vec(Edge_indx,:) = 0;
dAidy_vec(Edge_indx,:) = 0;

Grad_Ai_vec(:,1,:) = dAidx_vec(pixelList,:);
Grad_Ai_vec(:,2,:) = dAidy_vec(pixelList,:);

%
SFP_positions = [X(pixelList) Y(pixelList)]';

% (*) REDUCE DIMENSIONALITY OF MODEL FOR SPEED
Sj = Sj(:,:,1:num_p);
Sj_vec = Sj_vec(:,1:num_p);
Ai_vec = Ai_vec(:,1:num_lambda);
Grad_Ai_vec = Grad_Ai_vec(:,:,1:num_lambda);

% (4) evaluate the Jacobians
[dWdp, dNdq] = LK_warpJacobian(X, Y, pixelList, s0, Sj, Sja);

% can no longer precompute SD and H
%{
% (5) compute modified steepest descent images
SD = LK_SDimage(Grad_A0_vec, dNdq, dWdp, Ai_vec);

% (6) compute Hessian and inverse Hessian
H = LK_Hessian(SD);
inv_H = H^-1;
%}

% load raw data
[imageStack,sizeC,sizeZ,sizeT] = bf_openStack(fname_image);
imageStack = double(imageStack);

% reset starting point                                                          ********************
imageStack = imageStack(:,:,1:last_frame);

imageStack_vec = reshape(imageStack, [prod([size(imageStack,1),size(imageStack,2)]), size(imageStack,3)]);

% normalize the graylevel values in the image

% translate imageStack vectors to be zero mean
imageStack_vec = imageStack_vec - repmat(mean(imageStack_vec,1), [size(imageStack_vec,1),1]);

% make raw image vector one-variance !!!!!!!!!!!!!!!!!!!!
imageStack_vec = imageStack_vec ./ repmat(sqrt(sum(imageStack_vec.^2,1)), [size(imageStack_vec,1),1]);
imageStack = reshape(imageStack_vec, size(imageStack));

% invert imageStack time, to track backwards
Trev_imageStack = flipdim(imageStack,3);
