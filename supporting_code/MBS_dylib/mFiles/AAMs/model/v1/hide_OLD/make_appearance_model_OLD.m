function [mean_image_vector, R, Psi, Lambda, PsiT, X, Y, CC] = make_appearance_model(imageStack, dataSet, xbar_vector);

%{
notes:
this function assumes that the dataSet has already been corrected for zero-indexing in Fiji (by adding one to all positions values)
dataSet should also contain the somite number in column 7 of the data table
%}

% fixed parameters -------------------------------------------->

% used in "open and dilate mask"

openRadius = 8;
dilRadius  = 14;

% calculate a few useful things ------------------------------>

som_nums = unique(dataSet(:,7));
numpoints = size(unique(dataSet(:,1)), 1);
xbar_mat = [xbar_vector(1:numpoints), xbar_vector(numpoints+1:2*numpoints)];

% build shape normalized space ------------------------------->

% find pixels lying within mean shape

[X Y] = meshgrid([-1 : 0.01 : 1]);
mask = inpolygon(X, Y, [xbar_vector(1:numpoints); xbar_vector(1)], [xbar_vector(numpoints+1:2*numpoints); xbar_vector(numpoints+1)]);
mask = reshape(mask, size(X));

% open and dilate mask

nhood = fspecial('disk', openRadius);
nhood = ceil(nhood);
mask = imopen(mask,nhood);

nhood = fspecial('disk', dilRadius);
nhood = ceil(nhood);
mask = imdilate(mask,nhood);

clear nhood

% find list of points in mask

CC = bwconncomp(mask);
sampled_positions = [X(CC.PixelIdxList{1,1}) Y(CC.PixelIdxList{1,1})];

% use Thin-Plate Spline to warp somite appearance onto Shape-Free-Patch ------------------------------->

[X_image, Y_image] = meshgrid([1:size(imageStack(:,:,1),2)], [1:size(imageStack(:,:,1),1)]); 

k = 1;

for i = 1:length(som_nums),
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    frame_nums = unique(dataSet(som_indx,4));
    
    for j = 1:length(frame_nums),
        
        % read shape_vector out of dataSet
        
        frame_indx = find(dataSet(som_indx,4) == frame_nums(j));
        shape_mat = dataSet(som_indx(frame_indx), 2:3);
        
        % calculate warp function using Matlab's thin plate spline
        
        st = tpaps(xbar_mat', shape_mat', 1);
        warped_positions = fnval(st, sampled_positions')';
        
        % map image at warped_positions back into warped_image
        
        raw_image = imageStack(:,:,frame_nums(j));
        warped_image = zeros(size(X,1), size(X,2));
        
        for m = 1:length(CC.PixelIdxList{1,1}),
            
            warped_image(CC.PixelIdxList{1,1}(m)) = bilinear_interp(raw_image, warped_positions(m,2),...
            warped_positions(m,1));
        
        end
        
        % create vectorized image for PCA
        
        warped_image_vector = warped_image(CC.PixelIdxList{1,1});
        warped_image_vector = warped_image_vector(:);
        
        % store everything
        
        g(:,k) = warped_image_vector;
        
        k = k+1;
        
    end
    
end

% process image vector -------------------------------------->

% normalize g-vectors.  choose 'middle' vector as initial reference for alignment

mean_indx = ceil(size(g,2)/2);

[g, gbar, D] = normalize_image_vectors(g, mean_indx);

% do PCA

[mean_image_vector, R, Psi, Lambda, PsiT] = myPCA(g);