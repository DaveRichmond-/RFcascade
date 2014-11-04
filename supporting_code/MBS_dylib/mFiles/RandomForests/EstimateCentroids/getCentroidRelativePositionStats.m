%% get stats on centroid positions for MLE Gaussian Model

load('/Users/richmond/Data/Somites/ModelForSmoothing/centroids_forStats/gtCentroids.mat');

centroids = train_gtCentroid;

clear train_gtCentroid;
clear test_gtCentroid;

diff_centroids = diff(centroids, 1, 1);

dist_centroids = sqrt(sum(diff_centroids.^2,2));

mu = mean(dist_centroids, 3);
s2 = var(dist_centroids, 0, 3);

save('/Users/richmond/Data/Somites/ModelForSmoothing/centroids_forStats/centroidPositionStats.mat', 'mu', 's2');

% clear all