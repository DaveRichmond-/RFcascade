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

%% quick test

[fname_list] = getFileNames(pwd, '.tif');
for i = 1:12
    fname_list_train{i} = fname_list{2*i-1};
end

for i = 1:length(fname_list_train)
    figure,
    imagesc(imread(fname_list_train{i}))
    colormap('gray')
    hold on
    plot(train_gtCentroid(:,1,i),train_gtCentroid(:,2,i),'ro')
end