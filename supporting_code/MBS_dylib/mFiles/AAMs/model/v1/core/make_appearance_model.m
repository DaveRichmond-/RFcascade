function [mean_image_vector, R, Psi, Lambda, PsiT, X, Y, CC] = make_appearance_model(imageStack, dataSet, xbar_vector, SFPpoints);

%{
notes:
this function assumes that the dataSet has already been corrected for zero-indexing in Fiji (by adding one to all positions values)
dataSet should also contain the somite number in column 7 of the data table
%}

% calculate a few useful things ------------------------------>

som_nums = unique(dataSet(:,7));
allPoints = unique(dataSet(:,1));
numpoints = size(allPoints, 1);
xbar_mat = [xbar_vector(1:numpoints), xbar_vector(numpoints+1:2*numpoints)];

% build shape normalized space ------------------------------->

% find pixels lying within mean shape

[X Y] = meshgrid([-1 : 0.01 : 1]);

mask = make_mask(X, Y, xbar_vector, SFPpoints, allPoints);

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
        
        st = tpaps(xbar_mat', shape_mat', 1);     % uses ALL points, independent of sampled positions
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

% translate all vectors to be zero mean

g = g - repmat(mean(g,1), [size(g,1),1]);

% INSTEAD OF ALIGNING ALL IMAGE VECTORS, SIMPLY make all image vectors one-norm !!!!!!!!!!!!!!!!!!!!

g = g ./ repmat(sqrt(sum(g.^2,1)), [size(g,1),1]);

% choose 'middle' vector as reference and align all vectors to it
%{
mean_indx = ceil(size(g,2)/2);
[g, gbar, D] = normalize_image_vectors(g, mean_indx);
%}
% do PCA

[mean_image_vector, R, Psi, Lambda, PsiT] = myPCA(g);