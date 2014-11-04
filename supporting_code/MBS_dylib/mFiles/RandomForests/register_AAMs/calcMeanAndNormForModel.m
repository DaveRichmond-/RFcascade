%% calc mean and norm of image for normalizing image patches in grad descent

cd('/Users/richmond/Data/Somites/Processed/First set/Processing/GT/grayscale_images');

%{
plan
- iterate over all images
- calc mask using NearOptimal...
- pull out foreground into vector
- convert from int to double
- calc Mean of each image
- remove mean
- calculate norm of each image
- return average mean and norm for model
- how variable is this?


alternatively:
- threshold new image (to get rid of "corner effects") by t = 1
- calc mean, remove it, calc norm

%}

