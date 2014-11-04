function [mask, LoG_Image] = maskEmbryo(image, numThreshLevels, dilateRadius)

% calculates mask of embryo using near-optimal thresholding (Ambuhl, Sbalzarini)
% Things to consider changing:
% 1) Method of finding edge image.  Currently LoG, could use Grad, Var, etc.  Also, use abs() to deal with double edges.
% 2) Parameters: sigma = 3 for the LoG edge detector

% variables not passed to function

sigma = 3;      % radius of Gaussian blur for LoG filter

% convert image to double, if not already

image = double(image);

% calculate edge map using LoG -------------------------->

% specify filter

hsize = ceil(6*sigma);
hsize = floor(hsize/2)*2 + 1;       % ensures that filter size is ODD
h = fspecial('log', hsize, sigma);

LoG_Image = imfilter(image, h, 'symmetric', 'same');

% suppress double edges and then normalize

edgeImage = abs(LoG_Image);                             % CONSIDER JUST THRESHOLDING AT ZERO
min_edgeImage = min(min(edgeImage));
max_edgeImage = max(max(edgeImage));
edgeImage = (edgeImage - min_edgeImage)/(max_edgeImage - min_edgeImage);

% find optimal threshold -------------------------------->

threshLevels = linspace(0,1,numThreshLevels);
f_e = zeros(1,numThreshLevels);
f_f = zeros(1,numThreshLevels);
numPixels = size(image,1)*size(image,2);

for t=1:numThreshLevels,
    
    % apply thresh
    
    threshImage = edgeImage >= threshLevels(t);
    threshImage(:,1) = 0;
    threshImage(:,end) = 0;
    threshImage(1,:) = 0;
    threshImage(end,:) = 0;
    
    % filled image
    
    fillImage = imfill(threshImage,'holes');
    
    % calcalate fraction of 'active' pixels in edge and fill maps
    
    f_e(t) = sum(sum(threshImage))/numPixels;
    f_f(t) = sum(sum(fillImage))/numPixels;
    
    % store binary images
    %{
    threshALL(:,:,t) = threshImage;
    fillALL(:,:,t) = fillImage;
    %}
    
end

delta = f_f - f_e;

% find optimal threshold

[maxDiff indxDiff] = max(-diff(delta));
optThresh = threshLevels(indxDiff);

% build mask using optimal threshold -------------------->

optThreshImage = edgeImage >= optThresh;
optThreshImage(:,1) = 0;
optThreshImage(:,end) = 0;
optThreshImage(1,:) = 0;
optThreshImage(end,:) = 0;
optThreshImage = imfill(optThreshImage,'holes');

% find LARGEST connected component

CC = bwconncomp(optThreshImage,8);

for labelNum = 1:CC.NumObjects,
    componentSize(labelNum) = length(CC.PixelIdxList{1,labelNum});
end

[max_componentSize max_componentLabel] = max(componentSize);

% build mask from optThreshImage

mask = zeros(size(optThreshImage,1)*size(optThreshImage,2),1);
mask(CC.PixelIdxList{1,max_componentLabel}) = 1;
mask = reshape(mask,size(optThreshImage,1),size(optThreshImage,2));

% dilate mask to avoid clipping embryo boundaries

nhood = fspecial('disk',dilateRadius);
nhood = ceil(nhood);
mask = imdilate(mask,nhood);

% make sure no holes in the mask
mask = imfill(mask,'holes');