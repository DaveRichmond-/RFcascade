function [unaryFactors, pairwiseFactors, fitMasks] = AAM_perSomite_Inf_2inits(modelPath, grayImage, grayImageShape, probMap, probMapShape, sampling, numGDsteps, priorStrength, numOffsets, offsetScale, lambdaU, lambdaPW)

% function to export in shared library for use by c++ code
% does model based smoothing using AAM

%
display('launched AAM_perSomite_Inf_2inits')

% hard-code output flag for now, because varargin doesn't seem to work with calls to shared library
output_flag = 0;

% if nargin >= 11
%     output_flag = varargin{1};
% else
%     output_flag = 0;
% end

% prep ----------------------------------->

% load model
load(strcat(modelPath,'/modelForSmoothing.mat'));
% load('/Users/richmond/Data/Somites/ModelForSmoothing/centroidPositionStats.mat');

% reshape grayImage and probMap (passed in as row vector)
grayImage = reshape(grayImage, [grayImageShape(1), grayImageShape(2)]);
probMap = reshape(probMap, [probMapShape(1), probMapShape(2), probMapShape(3)]);

% upsample probMap to full res
if (sampling ~= 1)
    fullProbMap = zeros(probMapShape(1)*sampling, probMapShape(2)*sampling, probMapShape(3));
    for i = 1:size(probMap,3),
        fullProbMap(:,:,i) = upsample_noFilt(probMap(:,:,i), sampling);
    end
    fullProbMap = fullProbMap(1:grayImageShape(1), 1:grayImageShape(2), :);
    probMap = fullProbMap;
    clear fullProbMap
end

% visualize to check for permutation problems
if output_flag,
    figure,
    imagesc(grayImage)
end

% preprocess grayImage for fitting AAM
[grayImage] = preprocessGrayImage(grayImage, modelSegmentsAAM, output_flag);

% generate model fits ----------------------------------->

[segmentsFit, costs] = fitAAMtoProbMap_AAMperSomite(grayImage, probMap, modelCentroids, modelParams, modelSegmentsAAM, numGDsteps, priorStrength, numOffsets, offsetScale, output_flag);

[unaryFactors, pairwiseFactors, fitMasks] = factorsFromFits(segmentsFit, costs, lambdaU, lambdaPW, probMap, centroidStats);

% etch fitMasks, as was done to generate the gt data
uncertainty_radius = 2;
open_radius = 8;
[fitMasks] = etchFitMasks(fitMasks, uncertainty_radius, open_radius);

% resample fitMasks
if (sampling ~= 1)
    fitMasks = permute(downsample(permute(downsample(fitMasks, sampling),[2,1,3]), sampling),[2,1,3]);
end

% reshape output for VIGRA
fitMasks        = int8(reshape(fitMasks,[1, prod([size(fitMasks)])]));
unaryFactors    = reshape(unaryFactors,[1, prod([size(unaryFactors)])]);
pairwiseFactors = reshape(pairwiseFactors,[1, prod([size(pairwiseFactors)])]);