function [segmentsFit, costs] = fitAAMtoProbMap_AAMperSomite(grayImage, probMap, modelCentroids, modelParams, modelSegmentsAAM, numGDsteps, priorStrength, numOffsets, offsetScale, output_flag)

% model parameters that i don't want to change (for now)
num_lambda = 1;

% some user defined parameters.  these are hard-coded for now, because they work pretty well.
binSize = floor(size(probMap,1)/4);
lambda = floor(size(probMap,1)/16);
sigma = floor(size(probMap,1)/8);
% gridSpacing = 20;   % XY offsets of initialization for each segment

% useful variables
num_classes = size(probMap,3);

% 1) calc centroids for every class
centroids = zeros(num_classes-1,2); 
for i = 1:num_classes-1,
    centroids(i,:) = findCentroidFromProbMap(probMap(:,:,i+1), binSize, lambda, sigma, 0);
end

% visualize centroids
if output_flag,
    figure
    imagesc(probMap(:,:,1))
    hold on
    plot(centroids(:,1), centroids(:,2),'ko','MarkerSize',10)
end

% initialize AAM instances ------------------->

% using backbone model

[p_init, q_init, q_init2, Offsets1, Offsets2] = initializeModelSegmentsAAM_2inits_perSomite(centroids, modelCentroids, modelParams, modelSegmentsAAM, numOffsets, offsetScale);
L_init = zeros(num_lambda,size(p_init,2));

% fit AAM instances to grayscale image ------------------------->

LKparams.reg_weights = priorStrength;
LKparams.step_size   = 0.5*ones(6,1);
LKparams.num_iters   = numGDsteps;
LKparams.conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005; 0.0002];

% initialize
costs = zeros(size(probMap,3)-1, 2*numOffsets);
segmentsFit = zeros(2, 8, size(probMap,3)-1, 2*numOffsets);

for i = 1:size(costs,1),
    
    for j = 1:numOffsets,
        
        dq1 = [0; 0; Offsets1(i,1,j); Offsets1(i,2,j)] .* modelSegmentsAAM(i).weights;
        dq2 = [0; 0; Offsets2(i,1,j); Offsets2(i,2,j)] .* modelSegmentsAAM(i).weights;
                
        [segmentsFit(:,:,i,j), costs(i,j)]                       = LK_GradDescent_Sim_wPriors_forRF(grayImage, modelSegmentsAAM(i), q_init(:,i)  + dq1, p_init(:,i), L_init(:,i), LKparams);
        [segmentsFit(:,:,i,numOffsets+j), costs(i,numOffsets+j)] = LK_GradDescent_Sim_wPriors_forRF(grayImage, modelSegmentsAAM(i), q_init2(:,i) + dq2, p_init(:,i), L_init(:,i), LKparams);
        
    end
        
end