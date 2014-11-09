function [mu, s2] = getCentroidRelativePositionStats(dataPath, indx)

% get stats on centroid positions for MLE Gaussian Model

% load all data for models
load(strcat(dataPath, '/allSomitePositions.mat'));

% select out data to build model
posForStats = pos(:,:,:,indx);

centers = zeros(size(posForStats,3), 2, size(posForStats,4));
for j = 1:size(posForStats,4)
    
    for i = 1:size(posForStats,3)
        
        centers(i,:,j) = squeeze((posForStats(:,4,i,j) + posForStats(:,8,i,j)) / 2);

    end
end

diff_centers = diff(centers, 1, 1);

dist_centers = squeeze(sqrt(sum(diff_centers.^2,2)));

mu = mean(dist_centers, 2);
s2 = var(dist_centers, 0, 2);



% old solution, but uses CoM rather than center between landmarks 4 and 8.  not consistent with distance calculated from FitMasks.
%{
% load all data for models
load(strcat(dataPath, '/dataForBBM.mat'));

% select out data to build model
centroidsForStats = allCentroids(:,:,indx);

diff_centroids = diff(centroidsForStats, 1, 1);

dist_centroids = squeeze(sqrt(sum(diff_centroids.^2,2)));

mu = mean(dist_centroids, 2);
s2 = var(dist_centroids, 0, 2);

%}