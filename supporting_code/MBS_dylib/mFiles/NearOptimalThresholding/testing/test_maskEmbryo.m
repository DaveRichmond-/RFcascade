% test mask of embryo

%% clear workspace
clear

%% set options
options.fname = 'Focused 120807_bf_f0003_frame103.tif';
options.sigma = 10;
options.plotEdges = 1;

%% load image
im = bf_openStack(options.fname);
im = double(im);

%% variance map

% is normalization necessary?

%{
% specify circular neighborhood over which to calculate the variance
radius = 20;
nhood = fspecial('disk',radius);
nhood = ceil(nhood);

varImage = stdfilt(image,nhood);
figure
imagesc(varImage),
colormap('gray'),

% assign to edgeImage
edgeImage = varImage;
%}

%% edge map

% specify filter
hsize = ceil(6*options.sigma);
hsize = floor(hsize/2)*2 + 1;       % ensures that filter size is ODD
h1 = fspecial('log', hsize, options.sigma);
h2 = fspecial('laplacian', 0.2);
h3 = fspecial('sobel');

% detect edges
LoG_Image = imfilter(im, h1, 'symmetric', 'same');
Lap_Image = imfilter(im, h2, 'symmetric', 'same');
SobH_Image = imfilter(im, h3, 'symmetric', 'same');
SobV_Image = imfilter(im, h3', 'symmetric', 'same');


% plot
if options.plotEdges,
    figure, imagesc(LoG_Image); colormap('gray');
    %figure, imagesc(Lap_Image); colormap('gray');
    %figure, imagesc(sqrt(SobH_Image.^2 + SobV_Image.^2)); colormap('gray');
end

% assign to edgeImage and then abs and normalize
edgeImage = LoG_Image;
edgeImage = abs(edgeImage);            % CONSIDER JUST THRESHOLDING AT ZERO
edgeImage = (edgeImage-min(min(edgeImage)))/(max(max(edgeImage)) - min(min(edgeImage)));

figure,
imagesc(edgeImage),
colormap('gray')

%%

%figure
%imagesc(edgeImage), colormap('gray'),

% create threshold values
threshLevels = linspace(0,1,101);       % assumes normalized image !!!
numPixels = size(edgeImage,1)*size(edgeImage,2);

for t=1:length(threshLevels),
    
    % apply thresh blacken out the edges before fill
    threshImage = edgeImage >= threshLevels(t);
    threshImage(:,1) = 0;
    threshImage(:,end) = 0;
    threshImage(1,:) = 0;
    threshImage(end,:) = 0;
    f_e(t) = sum(sum(threshImage))/numPixels;
    
    % filled image
    fillImage = imfill(threshImage,'holes');
    f_f(t) = sum(sum(fillImage))/numPixels;
    
    % store
    threshALL(:,:,t) = threshImage;
    fillALL(:,:,t) = fillImage;
    
end

% diff
delta = f_f - f_e;

figure, hold on,
plot(threshLevels,f_f,'b')
plot(threshLevels,f_e,'g')
plot(threshLevels,delta,'r')
legend('f_f','f_e','delta')

% assign optimal threshold
[maxDiff indxDiff] = max(-diff(delta));
optThresh = threshLevels(indxDiff);

% build mask using optimal threshold
    
% apply thresh blacken out the edges before fill
mask = edgeImage >= optThresh;
mask(:,1) = 0;
mask(:,end) = 0;
mask(1,:) = 0;
mask(end,:) = 0;
mask = imfill(mask,'holes');

dilateRadius = 5;
nhood = fspecial('disk',dilateRadius);
nhood = ceil(nhood);
mask = imdilate(mask,nhood);

figure,
imagesc(mask)

%% test optThresh
for i=0:1,
    
    % apply thresh blacken out the edges before fill
    threshImage = edgeImage >= optThresh + i*(threshLevels(2)-threshLevels(1));
    threshImage(:,1) = 0;
    threshImage(:,end) = 0;
    threshImage(1,:) = 0;
    threshImage(end,:) = 0;
    temp_f_e = sum(sum(threshImage))/numPixels;
    
    % filled image
    fillImage = imfill(threshImage,'holes');
    temp_f_f = sum(sum(fillImage))/numPixels;
    
    %{
    figure
    colormap('gray')
    imagesc(threshImage)
    figure
    colormap('gray')
    imagesc(fillImage)
    %}
    
end

%{
% display
figure(1),
imagesc(edgeImage), 
colormap('gray'),

figure(2),
imagesc(threshImage), 
colormap('gray'),

figure(3),
imagesc(fillImage), 
colormap('gray'),
%}

%% segment embryo

%[labels, num_components] = bwlabel(mask, 8);
CC = bwconncomp(mask,8);

for labelNum = 1:CC.NumObjects,
    componentSize(labelNum) = length(CC.PixelIdxList{1,labelNum});
end

[max_componentSize max_componentLabel] = max(componentSize);

embryoMask = zeros(size(mask,1)*size(mask,2),1);
embryoMask(CC.PixelIdxList{1,max_componentLabel}) = 1;
embryoMask = reshape(embryoMask,size(mask,1),size(mask,2));

% return tight segmentation
segMask = embryoMask;
erodeRadius = 5;
nhood = fspecial('disk',erodeRadius);
nhood = ceil(nhood);
segMask = imerode(segMask, nhood);
segMask = uint8(segMask);

% dilate mask to avoid clipping embryo boundaries
dilateRadius = 30;
nhood = fspecial('disk',dilateRadius);
nhood = ceil(nhood);
embryoMask = imdilate(embryoMask,nhood);

figure
colormap('gray')
imagesc(segMask)