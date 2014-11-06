function [rigidModel] = buildRigidBackboneModel(modelPath, fname_list)

% modelPath: specifies path to rigidBackboneModel.mat, which stores centroids positions of all gt data, 
% as well as the corresponding filenames that each comes from

% load all data for models, and complete list of corresponding images
load(strcat(modelPath, '/rigidBackboneModel.mat'));

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
centroidsForModel = allCentroids(:,:,indx);

if (length(indx) == 1)

    % just return single instance
    rigidModel = squeeze(centroidsForModel);
    
else
    
    % align centroids
    cFMvec = reshape(centroidsForModel, [size(centroidsForModel,1)*size(centroidsForModel,2), size(centroidsForModel,3)]);
    L = norm(cFMvec(:,1));
    [~, mean_cFMvec, ~] = normalize_shape_vectors(cFMvec, 2);
    
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