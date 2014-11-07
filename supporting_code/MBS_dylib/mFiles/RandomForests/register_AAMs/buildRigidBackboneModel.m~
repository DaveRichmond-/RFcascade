function [rigidModel] = buildRigidBackboneModel(dataPath, fname_list)

% modelPath: specifies path to rigidBackboneModel.mat, which stores centroids positions of all gt data, 
% as well as the corresponding filenames that each comes from

% load all data for models, and complete list of corresponding images
load(strcat(dataPath, '/dataForBBM.mat'));

%
indx = findIndices(dataPath, fname_list);

% select out data to build model
centroidsForModel = allCentroids(:,:,indx);

if (length(indx) == 1)

    % just return single instance
    rigidModel = squeeze(centroidsForModel);
    
else
    
    % align centroids
    cFMvec = reshape(centroidsForModel, [size(centroidsForModel,1)*size(centroidsForModel,2), size(centroidsForModel,3)]);
    L = norm(cFMvec(:,1));
    [~, mean_cFMvec, ~] = normalize_shape_vectors(cFMvec, 1);
    
    % rescale
    mean_cFMvec = (L/norm(mean_cFMvec))*mean_cFMvec;
    rigidModel = reshape(mean_cFMvec, [size(centroidsForModel,1), size(centroidsForModel,2)]);

end

% visualize
%{
figure
hold on
for i = 1:size(centroidsForModel,3)
    plot(centroidsForModel(:,1,i),1024-centroidsForModel(:,2,i),'ko')
end
plot(rigidModel(:,1),1024-rigidModel(:,2),'c.','MarkerSize',30)
axis([0 1024 0 1024])
%}