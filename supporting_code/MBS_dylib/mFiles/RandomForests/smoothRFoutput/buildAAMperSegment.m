function [modelSegmentsAAM] = buildAAMperSegment(dataPath, indx, marginType, num_p, num_lambda)

%

% BUILD SHAPE MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load all data for shape model, and complete list of corresponding images
load(strcat(dataPath, '/dataForShapeModel.mat'));

% select out data to build model
xvec = all_xvec(:,:,indx);
num_somites = size(xvec,2);

% make a shape model per segment
for i = 1:num_somites
    
    % iteratively align all vectors
    [xvec_normalized, ~, ~] = normalize_shape_vectors(squeeze(xvec(:,i,:)), 1);
    
    % PCA
    [shape(i).xbar_vec, shape(i).R, shape(i).Psi, shape(i).Lambda, ~] = myPCA(xvec_normalized);
    
end

% BUILD APPEARANCE MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load all data for appearance model, and complete list of corresponding images
if marginType == 1
    load(strcat(dataPath, '/dataForAppModel1.mat'));
elseif marginType == 2
    load(strcat(dataPath, '/dataForAppModel2.mat'));
elseif marginType == 3
    load(strcat(dataPath, '/dataForAppModel3.mat'));
end

for i = 1:num_somites
    % PCA on subset of data specified by index
    [appear(i).gbar, appear(i).R, appear(i).Psi, appear(i).Lambda, ~] = myPCA(all_g{i}(:,indx));
    appear(i).X = all_X{i};
    appear(i).Y = all_Y{i};
    appear(i).pixelList = pixelList{i};
end

% CONVERT EVERYTHING INTO A PROPER MODEL FOR FITTING
for i = 1:num_somites
    [modelSegmentsAAM(i).s0,modelSegmentsAAM(i).s0_vec,modelSegmentsAAM(i).Sj,modelSegmentsAAM(i).Sj_vec,...
        modelSegmentsAAM(i).Sja,modelSegmentsAAM(i).Sja_vec,modelSegmentsAAM(i).weights,modelSegmentsAAM(i).A0,...
        modelSegmentsAAM(i).A0_vec,modelSegmentsAAM(i).Ai,modelSegmentsAAM(i).Ai_vec,modelSegmentsAAM(i).Grad_A0_vec,...
        modelSegmentsAAM(i).Grad_Ai_vec,modelSegmentsAAM(i).dWdp,modelSegmentsAAM(i).dNdq,modelSegmentsAAM(i).SFP_positions]...
        = LK_precompute_Simultaneous_forRF(shape(i), appear(i), num_p, num_lambda);
end

% save(strcat(outputPath,'/modelSegmentsAAM.mat'),'modelSegmentsAAM')