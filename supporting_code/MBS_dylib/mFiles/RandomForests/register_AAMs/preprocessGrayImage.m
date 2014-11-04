function [grayImage] = preprocessGrayImage(grayImage, modelSegmentsAAM, output_flag)

% pre-process the grayscale image for fitting ---------------------->

grayImage = double(grayImage);

% note: set this to 100 steps for now, to be a bit faster
[fishMask, LoG_Image] = maskEmbryo(grayImage, 1e2, 1);
threshMask = grayImage > 0;
mask = fishMask .* threshMask;
clear fishMask threshMask

% visualize
if output_flag,
    figure,
    imagesc(mask)
end

% blur image
h = fspecial('gaussian',21,3);
grayImage = imfilter(grayImage, h, 'same');

% select foreground pixels for calculating stats, make zero mean and one norm (over size of shape-free-patch, ie template)
numer = sum(grayImage(:).*mask(:));
denom = sum(mask(:));
grayImage = grayImage - numer/denom;
numer = norm(grayImage(:).*mask(:));
grayImage = (grayImage / numer) * (denom / numel(modelSegmentsAAM.A0));     % note scaling of norm to smaller patch
clear numer denom