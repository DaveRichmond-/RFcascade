function [modelFit] = fitModelToProbMap(probMap, modelMap, modelCentroids, numCentroidsUsed)

% some user defined parameters.  these are hard-coded for now, because they work pretty well.
% num_modes = 1;
params.binSize = floor(size(probMap,1)/4);
params.lambdaCoarse = floor(size(probMap,1)/16);
params.sigmaCoarse = floor(size(probMap,1)/8);
params.lambdaFine = floor(size(probMap,1)/32);
params.sigmaFine = floor(size(probMap,1)/64);

% useful variables
num_classes = size(probMap,3);

% 1) calc centroids for every class

%{
centroidSet = [];
for i = 2:num_classes,
    centroidSet(:,:,i) = findCentroidsFromProbMap2(probMap(:,:,i), num_modes, params, 0);    
end
% first class (BG) has no meaningful centroid
centroidSet = centroidSet(:,:,2:end);
%}

% visualize
%{
for i = 1:num_classes-1,
    figure,
    imagesc(max(probMap(:,:,2:end),[],3));
    hold on,
    plot(centroidSet(:,1,i),centroidSet(:,2,i),'wo','MarkerSize',10)
end
%}

% 2) randomly select centroid from set at each position
%{
selectCentroids = randi(num_modes, num_classes-1, 1);
for i = 1:num_classes-1,
    centroids(i,:) = centroidSet(selectCentroids(i),:,i);
end
%}

for i = 2:num_classes,
    centroids(i,:) = findCentroidsFromProbMap2(probMap(:,:,i), 1, params, 0);
end
% first class (BG) has no meaningful centroid
centroids = centroids(2:end,:);

% select somites to register to
centroidNums = randperm(num_classes-1);
centroidNums = centroidNums(1:numCentroidsUsed);
centroidNums = centroidNums(:);

% read out xy coords
x_mov = modelCentroids(centroidNums,1);
y_mov = modelCentroids(centroidNums,2);
x_fix = centroids(centroidNums,1);
y_fix = centroids(centroidNums,2);

modelFit = landmark_based_reg(modelMap, x_mov, y_mov, x_fix, y_fix);

% visualize
if (0),
    figure,
    imagesc(probMap(:,:,1))
    hold on,
    plot(x_fix,y_fix,'wo')
    
    figure,
    imagesc(mean(probMap(:,:,2:2:end),3))
    hold on,
    plot(x_fix,y_fix,'wo')
    
    figure,
    imagesc(mean(probMap(:,:,3:2:end),3))
    hold on,
    plot(x_fix,y_fix,'wo')

    figure,
    imagesc(modelFit),
    hold on,
    plot(x_fix,y_fix,'wo')
end