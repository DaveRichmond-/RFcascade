function [im] = make_image_from_image_vector(im_vector,X,pixelList);

% raster image vector back into a real image over the domain X with pixel locations specified by pixelList

im = NaN(size(X,1), size(X,2));
im(pixelList) = im_vector;