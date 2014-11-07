function [s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,Ai,Ai_vec,Grad_A0_vec,Grad_Ai_vec,dWdp,dNdq,SFP_positions] = LK_precompute_Simultaneous_forRF(shape,appear,num_p,num_lambda)

%

% basic properties of shape
shape_points = 8;
shape_dim = 2;

% define characteristic matrices from model
s0 = reshape(shape.xbar_vec, [shape_points, shape_dim])';
Sj = permute(reshape(shape.Psi, [shape_points, shape_dim, size(shape.Psi,2)]), [2 1 3]);

% vector representations
s0_vec  = reshape(permute(s0, [2 1]), [shape_points*shape_dim 1]);
Sj_vec  = squeeze(reshape(permute(Sj, [2 1 3]), [shape_points*shape_dim, 1, size(shape.Psi,2)]));

% make corresponding e-vectors of Similarity Transform
[Sja, Sja_vec, weights] = LK_SimTransfBasis(s0);

% basic properties of appearance
appear_pixels = size(appear.Psi,1);
appear_dof = size(appear.Psi,2);

% construct template and eigenimages as 2D images
[A0, Ai] = mean_eigen_images(appear.gbar, appear.X, appear.pixelList, appear.Psi);

% (*) REDUCE DIMENSIONALITY OF MODEL FOR SPEED
Sj = Sj(:,:,1:num_p);
Sj_vec = Sj_vec(:,1:num_p);
Ai = Ai(:,:,1:num_lambda);
% Ai_vec = Ai_vec(:,1:num_lambda);
% Grad_Ai_vec = Grad_Ai_vec(:,:,1:num_lambda);

% (3) Evaluate gradient of template, A0, and eigen images, Ai ---------->

% calculate gradient
[dA0dx, dA0dy] = gradient(A0, appear.X(1,:), appear.Y(:,1));

% calculate indices into pixelList where elements are non-NaN, and then apply to pixelList
Grad_indx = find(~isnan(dA0dx(:)) .* ~isnan(dA0dy(:)));
% Edge_indx = appear.pixelList(find(1 - ismember(appear.pixelList,Grad_indx)));                                             % can speed this up.
Edge_indx = appear.pixelList(~ismember(appear.pixelList,Grad_indx));

% remove NaN's at edge of gradient image (replaced by zeros)
dA0dx(Edge_indx) = 0;
dA0dy(Edge_indx) = 0;

%
Grad_A0_vec(:,1) = dA0dx(appear.pixelList);
Grad_A0_vec(:,2) = dA0dy(appear.pixelList);

% apply this domain to all 2D images to create condensed vector form
A0_vec = A0(appear.pixelList);

% repeat for eigen images
Ai_vec = reshape(Ai,[prod([size(Ai,1),size(Ai,2)]), size(Ai,3)]);
Ai_vec = Ai_vec(appear.pixelList,:);

for i = 1:num_lambda
    [dAidx(:,:,i), dAidy(:,:,i)] = gradient(Ai(:,:,i), appear.X(1,:), appear.Y(:,1));
end

dAidx_vec = reshape(dAidx,[prod([size(dAidx,1),size(dAidx,2)]) size(dAidx,3)]);
dAidy_vec = reshape(dAidy,[prod([size(dAidy,1),size(dAidy,2)]) size(dAidy,3)]);

dAidx_vec(Edge_indx,:) = 0;
dAidy_vec(Edge_indx,:) = 0;

Grad_Ai_vec(:,1,:) = dAidx_vec(appear.pixelList,:);
Grad_Ai_vec(:,2,:) = dAidy_vec(appear.pixelList,:);

%
SFP_positions = [appear.X(appear.pixelList) appear.Y(appear.pixelList)]';

% (4) evaluate the Jacobians
[dWdp, dNdq] = LK_warpJacobian(appear.X, appear.Y, appear.pixelList, s0, Sj, Sja);