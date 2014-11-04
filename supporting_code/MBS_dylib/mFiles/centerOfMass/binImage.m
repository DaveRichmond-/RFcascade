function [M] = binImage(im, binSize)

% return a binned image

% expand image to integer # of bins
new_imsize = [ceil(size(im,1)/binSize)*binSize, ...
              ceil(size(im,2)/binSize)*binSize];
M = zeros(new_imsize);
M(1:size(im,1), 1:size(im,2)) = im;

% useful variables
p = binSize;
q = binSize;

[m,n] = size(M);

M = sum(reshape(M,p,[]), 1);
M = reshape(M,m/p,[]).'; %Note transpose

M = sum( reshape(M,q,[]) ,1);
M = reshape(M,n/q,[]).'; %Note transpose