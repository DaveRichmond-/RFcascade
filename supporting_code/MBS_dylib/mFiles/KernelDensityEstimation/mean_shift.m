function [mu] = mean_shift(probMap, xy_init, lambda, sigma);

%{
calculates the mean shift over a fixed grid of points.  intended for mode-finding applications.

probMap: matrix image of probability values
xy_init: xy-position of initial guess.  not in ij-coords.  should have size = [1,2]
lambda: window radius of points to include in mean calculation
sigma: bandwidth for Gaussian kernel

%}

% make meshgrid over region of interest
L_bndry = max([floor(xy_init(1)) - lambda; 1]);
R_bndry = min([floor(xy_init(1)) + lambda + 1; size(probMap,2)]);
T_bndry = max([floor(xy_init(2)) - lambda; 1]);
B_bndry = min([floor(xy_init(2)) + lambda + 1; size(probMap,1)]);

[xs, ys] = meshgrid([L_bndry:R_bndry], [T_bndry:B_bndry]);

% prob map defines weights
w = probMap([ys(1,1):ys(end,1)], [xs(1,1):xs(1,end)]);

% raster into vectors for convenience
w = w(:);
xs = xs(:);
ys = ys(:);

% calc distance to xy_init
dist_sq = sum((repmat(xy_init,[length(xs),1]) - [xs,ys]).^2,2);

% mask distance
mask = dist_sq < lambda^2;
mask = mask(:);

K = exp(-dist_sq/(2*(sigma^2)));
K = times(K(:),mask);

mu(1) = (times(K,w)' * xs) / (K'*w);
mu(2) = (times(K,w)' * ys) / (K'*w);