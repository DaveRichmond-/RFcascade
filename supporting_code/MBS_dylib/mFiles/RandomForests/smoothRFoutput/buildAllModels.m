function [] = buildAllModels(dataPath, fname_list, marginType, num_p, num_lambda, output_path)

%

display('building AAM model per somite')
display(strcat('data path: ',dataPath));

% pull date_pos out of input fname_list
for i = 1:length(fname_list),
    k1 = strfind(fname_list{i}, '120');
    k2 = strfind(fname_list{i}, 'f00');
    fname_list_DatePos{i} = strcat(fname_list{i}(k1:k1+5),'_',fname_list{i}(k2:k2+4));
end

%
indx = findIndices(dataPath, fname_list_DatePos);

%
display('list of files used to build model is:')
for i = 1:length(indx)
    display(fname_list{i})
end
%
display(strcat('output path: ',output_path));

% build rigid model
modelCentroids = buildRigidBackboneModel(dataPath, indx);

% get stats on centroid distances
[mu, s2] = getCentroidRelativePositionStats(dataPath, indx);
centroidStats.mu = mu;
centroidStats.s2 = s2;

% build AAM model
modelSegmentsAAM = buildAAMperSegment(dataPath, indx, marginType, num_p, num_lambda);

[q_model, p_model] = initializeAAMperSegment(dataPath, indx, modelCentroids, modelSegmentsAAM);

modelParams.q_model = q_model;
modelParams.p_model = p_model;

save(strcat(output_path,'/modelForSmoothing.mat'), 'modelCentroids', 'centroidStats', 'modelParams', 'modelSegmentsAAM')