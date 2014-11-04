function [sp_save, q, p, L, cost] = fitAAMtoProbMap_gridSampleDebug(grayImage, probMap, modelCentroids, modelSegmentsAAM, numGDsteps, varargin);

% model parameters that i don't want to change (for now)
num_p = 1;
num_lambda = 1;

% some user defined parameters.  these are hard-coded for now, because they work pretty well.
binSize = floor(size(probMap,1)/4);
lambda = floor(size(probMap,1)/16);
sigma = floor(size(probMap,1)/8);
gridSpacing = 30;   % XY offsets of initialization for each segment

% useful variables
num_classes = size(probMap,3);

% 1) calc centroids for every class
for i = 2:num_classes,
    centroids(i,:) = findCentroidFromProbMap(probMap(:,:,i), binSize, lambda, sigma, 0);
end
% first class (BG) has no meaningful centroid
centroids = centroids(2:end,:);

% initialize AAM instances ------------------->

numCentroidsUsed = size(centroids,1);   % use all centroids
[p_init, q_init] = initializeModelSegmentsAAM(centroids, modelCentroids, numCentroidsUsed, modelSegmentsAAM);
L_init = zeros(num_lambda,size(p_init,2));

% create grid of initialization positions
[gridX gridY] = meshgrid([-gridSpacing:gridSpacing:gridSpacing]);
gridX = gridX(:);
gridY = gridY(:);

% fit AAM instances to grayscale image ------------------------->

LKparams.reg_weights = 1e-3*[1; 1; 1e-1; 1e-1; 1e3; 0];
LKparams.step_size = 0.5*ones(6,1);
LKparams.num_iters = numGDsteps;
LKparams.conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005; 0.0002];

if nargin == 7,
    i1 = varargin{1};
    i2 = varargin{1};
    j1 = varargin{2};
    j2 = varargin{2};
else
    i1 = 1;
    i2 = size(probMap,3)-1;
    j1 = 1;
    j2 = length(gridX(:));
end

for i = i1:i2,
    
    i,
    
    for j = j1:j2,
        
        j,
              
        Offset = [0; 0; gridX(j); gridY(j)];

        tic,
        [q_final_save(:,i), p_final_save(:,i), L_final(:,i), sp_final_save(:,:,i), q, p, L, sp_save, conv_flag_save(i), cost] = LK_GradDescent_Sim_wPriors_forRFdebug(grayImage, modelSegmentsAAM, q_init(:,i) + Offset, p_init(:,i), L_init(:,i), q_init(:,i), p_init(:,i), L_init(:,i), LKparams);
        toc,
        
    end
end