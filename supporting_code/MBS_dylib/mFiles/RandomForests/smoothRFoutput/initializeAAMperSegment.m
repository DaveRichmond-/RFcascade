function [] = initializeAAMperSegment(dataPath, modelSegmentsAAM, modelCentroids, allCentroidsForModel)

%

% load all XY data
load(strcat(dataPath, '/allSomitePositions.mat'));

%
num_somites = size(modelSegmentsAAM,2);
num_instances = size(allCentroids,3);

% initialize
pos_registered = zeros(size(pos));

for k = 1:num_instances

    % read out xy coords
    x_mov = allCentroidsForModel(:,1,k);
    y_mov = allCentroidsForModel(:,2,k);
    x_fix = modelCentroids(:,1);
    y_fix = modelCentroids(:,2);
    
    % calc warp from centroids
    tform = fitgeotrans([x_mov, y_mov], [x_fix, y_fix], 'Similarity');
    tform = maketform('affine', tform.T);

    for i = 1:num_somites
        
        s_pos = squeeze(pos(:,:,i,k));
        
        [X, Y] = tformfwd(tform, s_pos(1,:), s_pos(2,:));
        pos_registered(:,:,i,k) = [X; Y];

    end
    
end

% average all instances of same somite
for i = 1:num_somites
    
    pos_registered_avg(:,:,i) = mean(pos_registered(:,:,i,:),4);
    
end

% back-calculate the parameters of this warp