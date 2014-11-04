function [smoothProbMap] = modelBasedSmoothing_wARGS(probMap, probMapShape, numCentroidsUsed, numFits, sampling)

% runs 'fitModelToProbMap' multiple times, to get a smooth 'model-based' representation of a probability distribution

figure,
title('launched!')
pause(5)

% load model
modelMap = imread('/Users/richmond/Data/Somites/ModelForSmoothing/label_registered22.tif');
load('/Users/richmond/Data/Somites/ModelForSmoothing/label_registered22_centroids.mat');

% reshape probMap (passed in as row vector)
probMap = probMap(:);
probMap = reshape(probMap, [probMapShape(1), probMapShape(2), probMapShape(3)]);
probMap = permute(probMap, [2, 1, 3]);

% resample model to correspond to input data
modelMap = downsample(modelMap, sampling);
modelMap = downsample(permute(modelMap, [2 1 3]), sampling);
modelMap = permute(modelMap, [2 1 3]);

% l = 0;
% k = 0;
% for i = 1:sampling:size(modelMap, 1),
%     k = k+1;
%     for j = 1:sampling:size(modelMap, 2),
%         l = l+1;
%         modelMap(k,l,:) = modelMap(i,j,:);
%     end
% end
% modelMap = modelMap(1:k,1:l,:);

% resize modelCentroids by sampling
modelCentroids = ((modelCentroids-1)/double(sampling)) + 1;

% test resampling
if (~ isequal(size(modelMap,1), size(probMap,1)) || ~ isequal(size(modelMap,2), size(probMap,2))),
    error('resampling failed')
end

% useful variables
num_classes = size(probMap,3);

% generate model fits
modelFits = zeros(size(probMap,1), size(probMap,2), numFits);
for i = 1:numFits,
    
    modelFits(:,:,i) = fitModelToProbMap(probMap, modelMap, modelCentroids, numCentroidsUsed);
    
end

% calculate weights for model fits
class_list = [0:num_classes-1];
weight = zeros(1,size(modelFits,3));

for j = 1:size(modelFits,3),

    for i = 1:num_classes;
        
        c = class_list(i);
        
        % calculate weight for particular instance
        mask = modelFits(:,:,j) == c;
        mask = mask(:);

        p = probMap(:,:,i);
        p = p(:);
        
        weight(j) = weight(j) + (p'*mask)/sum(mask);
        
    end

    weight(j) =  weight(j) / length(class_list);          % so that weights scale between [0,1]
    
end

%

for i = 1:num_classes;
    
    % 
    ClassMap = modelFits == class_list(i);

    for j = 1:size(modelFits,3);
        ClassMap(:,:,j) = ClassMap(:,:,j)*weight(j);
    end
    

    smoothProbMap(:,:,i) = sum(ClassMap,3); %/sum(weight,2);
    
%     figure,
%     imagesc(smoothProbMap(:,:,i))
%     colormap('gray')

end

smoothProbMap = smoothProbMap ./ repmat(sum(smoothProbMap,3),[1 1 num_classes]);