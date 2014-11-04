%% test fitModelToProbMap

%% load model

modelMap = imread('label_registered22.tif');
load ('label_registered22_centroids.mat');

%% load prob data to smooth

probMap = bf_openStack('image#7_probs_stack.tif');

%% test model fitting

%{
modelFit = fitModelToProbMap(probMap, modelMap, modelCentroids, 21);

% visualize

figure,
imagesc(modelFit),

% save

imwrite(modelFit, 'modelFit.tif')
%}

%%

numCentroidsUsed = 11;
numFits = 100;

smoothProbMap = modelBasedSmoothing(probMap, modelMap, modelCentroids, numCentroidsUsed, numFits);

%%

smoothProbMap = single(smoothProbMap);
bfsave(smoothProbMap, 'smoothProbMap_11ptFit.tif')

%%

figure,
imagesc(smoothProbMap(:,:,1)),
colormap('gray')

figure,
imagesc(probMap(:,:,1)),
colormap('gray')

%% export as executable

%% test executable first in matlab

clear all,
modelBasedSmoothing_exec;

%% export to standalone app

mcc -mv modelBasedSmoothing_exec.m -a label_registered22.tif -a label_registered22_centroids.mat % -a image#7_probs_stack.tif 

%% export to shared lib

mcc -v -W cpplib:libmodelBasedSmoothing2 -T link:lib modelBasedSmoothing_wARGS

%% build

mbuild RF_Cascade_wMBS_Learn.cpp libmodelBasedSmoothing2.dylib

%% test AAM model fit

below....


%% load model

% cd('/Users/richmond/Data/Somites/ModelForSmoothing')
% modelMap = imread('label_registered22.tif');
% load ('label_registered22_centroids.mat');
% load('modelSegmentsAAM.mat')

%%

clear all

% load test data
cd('/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing/MBS_AAMs/test_Matlab')
probMap = bf_openStack('probs_stack.tif');
grayImage = imread('/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing/MBS_AAMs/test_Matlab/grayscale_registered16.tif');

%% mimic c++ inputs

probMapShape = size(probMap);
% probMap = permute(probMap, [2 1 3]);
probMap = probMap(:);

grayImageShape = size(grayImage);
% grayImage = permute(grayImage, [2,1,3]);
grayImage = grayImage(:);

sampling = 1;

%% run

% other user set parameters
% numCentroidsUsed = 11;
% numFits = 50;
numGDsteps = 3;
lambdaU = 4;
lambdaPW = 4;

% old version - returns smoothProbMap
[smoothProbMap] = MBS_AAM_gS(grayImage, grayImageShape, probMap, probMapShape, sampling, numGDsteps, lambdaU);

% new version - returns factors
% [unaryFactors, pairwiseFactors, fitMasks] = MBS_AAM_forINF(grayImage, grayImageShape, probMap, probMapShape, sampling, numGDsteps, lambdaU, lambdaPW);
% [unaryFactors, pairwiseFactors, fitMasks] = AAM_Inf_2inits(grayImage, grayImageShape, probMap, probMapShape, sampling, numGDsteps, lambdaU, lambdaPW, 0);
% fitMasks = reshape(fitMasks, [grayImageShape(1),grayImageShape(2),18*(probMapShape(3)-1)]);

% thresh_mask = reshape(thresh_mask, [probMapShape(1), probMapShape(2), size(thresh_mask,2)]);

% weight_mask = reshape(weight_mask, [probMapShape(1), probMapShape(2), size(weight_mask,2)]);

%%

bfsave(int8(fitMasks), 'fitMasks.tif')

%%


for i = 1:9,%size(thresh_mask,3),
    figure,
    imagesc(weight_mask(:,:,i)),
    colormap('gray')
%     colorbar;
%     pause(1)
end

%%

% smoothProbMap = probOfFit(probMap, segmentsFit, costs, 10, 10);

for i = 1:size(smoothProbMap,3),
    figure,
    imagesc(smoothProbMap(:,:,i)),
    colorbar;
    pause(1)
end

%%

% smoothProbMap = single(smoothProbMap);
bfsave(smoothProbMap, 'smoothProbMap_lambda=2.tif')

%% export

mcc -v -W cpplib:libmodelBasedSmoothing2 -T link:lib modelBasedSmoothing_wARGS MBS_AAM_gS MBS_AAM_forINF AAM_Inf_2inits

%%

mbuild dummy.cpp libMBS_AAM_gS.dylib