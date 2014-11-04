function [im_up] = upsample_noFilt(im, sampling)

% resample image so that each pixel is duplicated "sampling" number of times in each dimension.
% assumes 2d image

%
im_up = zeros(size(im,1)*sampling, size(im,2)*sampling);

%
for i = 1:sampling,
    
    for j = 1:sampling,
        
        im_up([i:sampling:end],[j:sampling:end]) = im;
        
    end
    
end