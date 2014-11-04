function [segmentsFit, costs] = fitAAMtoProbMap_gridSample(grayImage, probMap, modelCentroids, modelSegmentsAAM, numGDsteps, output_flag);

% model parameters that i don't want to change (for now)
num_p = 1;
num_lambda = 1;

% some user defined parameters.  these are hard-coded for now, because they work pretty well.
binSize = floor(size(probMap,1)/4);
lambda = floor(size(probMap,1)/16);
sigma = floor(size(probMap,1)/8);
gridSpacing = 20;   % XY offsets of initialization for each segment

% useful variables
num_classes = size(probMap,3);

% 1) calc centroids for every class
for i = 2:num_classes,
    centroids(i,:) = findCentroidFromProbMap(probMap(:,:,i), binSize, lambda, sigma, 0);
end
% first class (BG) has no meaningful centroid
centroids = centroids(2:end,:);

% visualize centroids
if output_flag,
    figure
    imagesc(probMap(:,:,1))
    hold on
    plot(centroids(:,1), centroids(:,2),'ko','MarkerSize',10)
end

% initialize AAM instances ------------------->

numCentroidsUsed = size(centroids,1);   % use all centroids
[p_init, q_init] = initializeModelSegmentsAAM(centroids, modelCentroids, numCentroidsUsed, modelSegmentsAAM);
L_init = zeros(num_lambda,size(p_init,2));

% create grid of initialization positions
[gridX, gridY] = meshgrid([-gridSpacing:gridSpacing:gridSpacing]);
gridX = gridX(:);
gridY = gridY(:);

% fit AAM instances to grayscale image ------------------------->

LKparams.reg_weights = 1e-3*[1; 1; 1e-1; 1e-1; 1e3; 0];
LKparams.step_size = 0.5*ones(6,1);
LKparams.num_iters = numGDsteps;
LKparams.conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005; 0.0002];

% initialize
costs = zeros(size(probMap,3)-1, length(gridX(:)));
segmentsFit = zeros(2, 8, size(probMap,3)-1, length(gridX(:)));

for i = 1:size(probMap,3)-1,
    
%     display(strcat('segment# = ',num2str(i))),
    
    for j = 1:length(gridX(:)),
        
        Offset = [0; 0; gridX(j); gridY(j)];
        [segmentsFit(:,:,i,j), costs(i,j)] = LK_GradDescent_Sim_wPriors_forRF(grayImage, modelSegmentsAAM, q_init(:,i) + Offset, p_init(:,i), L_init(:,i), q_init(:,i), p_init(:,i), L_init(:,i), LKparams);
        
    end
end