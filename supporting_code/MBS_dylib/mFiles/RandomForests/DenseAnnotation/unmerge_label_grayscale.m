function [im, wm] = unmerge_label_grayscale(merged);

% function to merge label and grayscale images, using 'watermarking' technique
% pre-processing for joint rotation

% foce 8bit input images
im = round(merged/(2^8));
wm = merged - im*(2^8);

% recast
im = uint8(im);
wm = uint8(wm);