function modelBasedSmoothing_exec;

% runs 'fitModelToProbMap' multiple times, to get a smooth 'model-based' representation of a probability distribution

% load data
[fname_list] = getFileNames(pwd, 'image#7');
% probMap = bf_openStack('image#7_probs_stack.tif');
for i = 1:length(fname_list)
    probMap(:,:,i) = imread(fname_list{i});
end
modelMap = imread('label_registered22.tif');
load ('label_registered22_centroids.mat');

% hard-coded variables (to avoid passing arguments for now)
numCentroidsUsed = 11;
numFits = 100;

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

% output (for testing purposes)
figure,
imagesc(smoothProbMap(:,:,1)),