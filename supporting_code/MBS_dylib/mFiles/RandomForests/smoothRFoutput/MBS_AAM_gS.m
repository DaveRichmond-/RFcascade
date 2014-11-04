function [smoothProbMap] = MBS_AAM_gS(grayImage, grayImageShape, probMap, probMapShape, sampling, numGDsteps, lambda, varargin)

% function to export in shared library for use by c++ code
% does model based smoothing using AAM

%
display('launched MBS_AAM_gS')

if nargin >= 9
    output_flag = varargin{1};
else
    output_flag = 0;
end

% prep ----------------------------------->

% load
load('/Users/richmond/Data/Somites/ModelForSmoothing/label_registered22_centroids.mat');
load('/Users/richmond/Data/Somites/ModelForSmoothing/modelSegmentsAAM.mat');

% reshape grayImage (passed in as row vector)
grayImage = reshape(grayImage, [grayImageShape(1), grayImageShape(2)]);
probMap = reshape(probMap, [probMapShape(1), probMapShape(2), probMapShape(3)]);

% preprocess grayImage for fitting AAM
[grayImage] = preprocessGrayImage(grayImage, modelSegmentsAAM, output_flag);

% upsample probMap to full res
fullProbMap = zeros(probMapShape(1)*sampling, probMapShape(2)*sampling, probMapShape(3));
for i = 1:size(probMap,3),
    fullProbMap(:,:,i) = upsample_noFilt(probMap(:,:,i), sampling);
end
fullProbMap = fullProbMap(1:grayImageShape(1), 1:grayImageShape(2), :);
probMap = fullProbMap;
clear fullProbMap

% visualize to check for permutation problems
if output_flag,
    figure,
    imagesc(grayImage)
end

% generate model fits ----------------------------------->

% initialize

[segmentsFit, costs] = fitAAMtoProbMap_gridSample(grayImage, probMap, modelCentroids, modelSegmentsAAM, numGDsteps, output_flag);

[smoothProbMap] = probMapFromFits(probMap, segmentsFit, costs, lambda);

% resample smoothProbMap
smoothProbMap = permute(downsample(permute(downsample(smoothProbMap, sampling),[2,1,3]), sampling),[2,1,3]);