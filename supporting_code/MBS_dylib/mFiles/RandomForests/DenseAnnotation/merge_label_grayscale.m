function [merged] = merge_label_grayscale(im,wm);

% function to merge label and grayscale images, using 'watermarking' technique
% pre-processing for joint rotation
% im = image
% wm = watermark (gets overlaid)

% foce 8bit input images
wm = uint8(wm);
im = uint8(im);

% rescale im
im = uint16(im);
im = im*(2^8);

% recast wm
wm = uint16(wm);

% merge images
merged = im + wm;