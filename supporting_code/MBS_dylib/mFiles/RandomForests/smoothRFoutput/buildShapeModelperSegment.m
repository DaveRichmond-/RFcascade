function [ShapeModel] = buildShapeModelperSegment(modelPath, fname_list)

%

% BUILD SHAPE MODEL

% load all data for shape model, and complete list of corresponding images
load(strcat(modelPath, '/dataForShapeModel.mat'));

% find index of each element in fname_list
indx = zeros(1,length(fname_list));
for i = 1:length(fname_list)    
    tmp = find(strcmp(fname_list{i},fname_complete_list));
    if (isempty(tmp)),
        display(strcat(fname_list{i},' not found in complete list'))
        indx(i) = [];
    else
        indx(i) = tmp;
    end
end

% debug warning.
if (length(unique(indx)) ~= length(indx))
    display(strcat('repeated filename in "buildRigidBackboneModel.m".  Number of unique filenames is: ',num2str(length(unique(indx)))))
    indx = unique(indx);
end

% select out data to build model
xvec = all_xvec(:,:,indx);

% make a shape model per segment
for i = 1:size(xvec,2)
    
    % iteratively align all vectors
    [xvec_normalized, ~, ~] = normalize_shape_vectors(squeeze(xvec(:,i,:)), 1);
    
    % PCA
    [ShapeModel(i).xbar, ShapeModel(i).R, ShapeModel(i).Psi, ShapeModel(i).Lambda, ~] = myPCA(xvec_normalized);
    
end