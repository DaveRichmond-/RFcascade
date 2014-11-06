function [p_init, q_init, q_init2, Offsets1, Offsets2] = initializeModelSegmentsAAM_2inits(centroids, modelCentroids, modelSegmentsAAM, numOffsets, offsetScale)

% takes centroids calculated from RF output, and registers AAM instances, based on a centroid model
% then recenter each AAM instance, so that it's centroid perfectly matches the centroid of the RF output


% read out xy coords
x_mov = modelCentroids(:,1);
y_mov = modelCentroids(:,2);
x_fix = centroids(:,1);
y_fix = centroids(:,2);

% calc warp from centroids
tform = fitgeotrans([x_mov, y_mov], [x_fix, y_fix], 'Similarity');
tform = maketform('affine', tform.T);

% post-processing, calculate offset of centroid of interest from ideal position.  store this, and apply below.
[X, Y] = tformfwd(tform, modelCentroids(:,1), modelCentroids(:,2));
delta = centroids - [X, Y];

% unpack what you need from modelSegmentsAAM
s0 = modelSegmentsAAM.s0;
s0_vec = modelSegmentsAAM.s0_vec;
Sj = modelSegmentsAAM.Sj;
Sja = modelSegmentsAAM.Sja;
Sja_vec = modelSegmentsAAM.Sja_vec;
weights = modelSegmentsAAM.weights;
p_model = modelSegmentsAAM.p_model;
q_model = modelSegmentsAAM.q_model;

% initialize
p_init  = zeros(1, size(centroids,1));
q_init  = zeros(size(Sja,3), size(centroids,1));
q_init2 = zeros(size(Sja,3), size(centroids,1));

% generate instances of AAM and apply tform warp
for i = 1:size(centroids,1),
    
    % calc model instance
    s = s0 + Sj*p_model(i);
    [A,b] = LK_qtoA(q_model(:,i), weights);
    s = A*s + repmat(b, [1, size(s,2)]);
        
    % register to get initialization
    [X, Y] = tformfwd(tform, s(1,:), s(2,:));
    s_init = [X; Y];
    % apply correction (from above)
    s_init2 = s_init + repmat(delta(i,:)',[1, size(s_init,2)]);
    % vectorize
    s_init_vec  = reshape(permute(s_init, [2 1]),[numel(s_init), 1]);
    s_init_vec2 = reshape(permute(s_init2,[2 1]),[numel(s_init2),1]);

    % calc corresponding q_init
    for j = 1:size(Sja,3),
        q_init(j,i)  = dot(Sja_vec(:,j), (s_init_vec  - s0_vec));
        q_init2(j,i) = dot(Sja_vec(:,j), (s_init_vec2 - s0_vec));
    end
    p_init(i) = p_model(i);
    
    % visualize
%     xv = s_init(1,:)';
%     yv = s_init(2,:)';
%     [Xgrid, Ygrid] = meshgrid([1:1024],[1:1024]);
%     mask(:,:,i) = inpolygon(Xgrid, Ygrid, [xv; xv(1)], [yv; yv(1)]);
    
end

% visualize
% figure,
% imagesc(max(mask,[],3))
% pause(3)

% calculate Offsets along the line between neighboring segments, using registered model
[Xmodel, Ymodel] = tformfwd(tform, modelCentroids(:,1)', modelCentroids(:,2)');
registeredModelCentroids = [Xmodel(:), Ymodel(:)];
Offsets1 = lineOffsets(registeredModelCentroids, numOffsets, offsetScale);
% ... using centroids of RF output
Offsets2 = lineOffsets(centroids, numOffsets, offsetScale);