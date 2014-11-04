function [smoothProbMap] = MBS_AAM(grayImage, probMap, modelCentroids, modelSegmentsAAM, numCentroidsUsed, numFits, numGDsteps, lambda, BG);

% runs 'fitModelToProbMap' multiple times, to get a smooth 'model-based' representation of a probability distribution

% useful variables
num_classes = size(probMap,3);

% pre-process the grayscale image for fitting ---------------------->
grayImage = double(grayImage);

% blur image
h = fspecial('gaussian',51,3);
grayImage = imfilter(grayImage, h, 'same');

grayImage_vec = grayImage(:);

% translate imageStack vectors to be zero mean
grayImage_vec = grayImage_vec - mean(grayImage_vec);

% make raw image vector one-variance !!!!!!!!!!!!!!!!!!!!
grayImage_vec = grayImage_vec ./ sum(grayImage_vec.^2);               % old version actually used one-standard-deviation
grayImage = reshape(grayImage_vec, size(grayImage));

% generate model fits ----------------------------------->

weights = zeros(1,numFits);

segmentsFit = zeros(2,8,num_classes-1,numFits);
costs = zeros(num_classes-1,numFits);

for i = 1:numFits,
    
    [segmentsFit(:,:,:,i), costs(:,i)] = fitAAMtoProbMap(grayImage, probMap, modelCentroids, modelSegmentsAAM, numCentroidsUsed, numGDsteps);
    
end

smoothProbMap = probOfFit(probMap, segmentsFit, costs, lambda, BG);