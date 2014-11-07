function [q_init, p_init] = initializeAAMperSegment(dataPath, fname_list, modelCentroids, modelSegmentsAAM)

%

% load all XY data
load(strcat(dataPath, '/allSomitePositions.mat'));
load(strcat(dataPath, '/dataForInitialization.mat'));

%
indx = findIndices(dataPath, fname_list);
centroids = all_centroids_unregistered(:,:,indx);

%
num_somites = size(modelCentroids,1);
num_instances = size(centroids,3);

% initialize
pos_registered = zeros(size(pos,1),size(pos,2),num_somites,num_instances);

x_fix = modelCentroids(:,1);
y_fix = modelCentroids(:,2);

for k = 1:num_instances

    % read out xy coords
    x_mov = centroids(:,1,k);
    y_mov = centroids(:,2,k);
    
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

% visualize

figure,
hold on,
plot(modelCentroids(:,1),modelCentroids(:,2),'ro')
for i = 1:size(pos_registered_avg,3)
    plot(pos_registered_avg(1,:,i),pos_registered_avg(2,:,i))
end
axis([0 1024 0 1024])

% back-calculate the parameters of this warp
q_init = zeros(4, num_somites);
p_init = zeros(1,num_somites);
for i = 1:num_somites
    % calc the transformation
    pos_reg_avg_vec = reshape(permute(pos_registered_avg(:,:,i),[2,1,3]),[size(pos_registered_avg,1)*size(pos_registered_avg,2),1]);
    for j = 1:4
        q_init(j,i) = dot(modelSegmentsAAM(i).Sja_vec(:,j), (pos_reg_avg_vec - modelSegmentsAAM(i).s0_vec));
    end
    % start at the mean shape
    p_init(i) = 0;
end

% visualize forward projection
figure,
hold on,
plot(modelCentroids(:,1),modelCentroids(:,2),'ro')
for i = 1:21,
    
    % unpack what you need from modelSegmentsAAM
    s0 = modelSegmentsAAM(i).s0;
%     s0_vec = modelSegmentsAAM(i).s0_vec;
    Sj = modelSegmentsAAM(i).Sj;
%     Sja = modelSegmentsAAM(i).Sja;
%     Sja_vec = modelSegmentsAAM(i).Sja_vec;
    weights = modelSegmentsAAM(i).weights;

    % calc model instance
    s = s0 + Sj*p_init(i);
    [A,b] = LK_qtoA(q_init(:,i), weights);
    s = A*s + repmat(b, [1, size(s,2)]);

    plot(s(1,:),s(2,:))
    
end
axis([0 1024 0 1024])