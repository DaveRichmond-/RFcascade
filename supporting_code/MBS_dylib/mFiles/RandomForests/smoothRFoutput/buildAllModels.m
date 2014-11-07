function [] = buildAllModels(dataPath, fname_list, marginType, num_p, num_lambda, output_path)

%

% pull date_pos out of input fname_list
for i = 1:length(fname_list),
    k1 = strfind(fname_list{i}, '120');
    k2 = strfind(fname_list{i}, 'f00');
    fname_list{i} = strcat(fname_list{i}(k1:k1+5),'_',fname_list{i}(k2:k2+4));
end

% build rigid model
[modelCentroids, allCentroidsForModel] = buildRigidBackboneModel(dataPath, fname_list);

% build AAM model
modelSegmentsAAM = buildAAMperSegment(dataPath, fname_list, marginType, num_p, num_lambda);

initializeAAMperSegment(dataPath, modelSegmentsAAM, modelCentroids, allCentroidsForModel);

save(strcat(output_path,'/modelForSmoothing.mat'), 'modelCentroids', 'modelSegmentsAAM')