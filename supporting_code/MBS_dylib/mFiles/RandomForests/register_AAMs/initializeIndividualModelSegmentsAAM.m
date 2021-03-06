function [p_init, q_init] = initializeIndividualModelSegmentsAAM(centroids, modelCentroids, modelSegmentsAAM)

% takes centroids calculated from RF output, and registers AAM instances, using nearest neighbors to specify rotation



% unpack what you need from modelSegmentsAAM
s0 = modelSegmentsAAM.s0;
s0_vec = modelSegmentsAAM.s0_vec;
Sj = modelSegmentsAAM.Sj;
Sja = modelSegmentsAAM.Sja;
Sja_vec = modelSegmentsAAM.Sja_vec;
weights = modelSegmentsAAM.weights;
p_model = modelSegmentsAAM.p_model;
q_model = modelSegmentsAAM.q_model;

% generate instances of AAM and apply tform warp
for i = 1:size(centroids,1),

    % FIRST, DEFINE WARP FROM CENTROID OF INTEREST, AND NEIGHBOR ------------------->

    % select somites to register to
    if i == 1,
        centroidNums = [1;2;3];
    elseif i == size(centroids,1),
        centroidNums = size(centroids,1) - [2;1;0];
    else
        centroidNums = [i-1; i; i+1];
    end
    
    % read out xy coords
    x_mov = modelCentroids(centroidNums,1);
    y_mov = modelCentroids(centroidNums,2);
    x_fix = centroids(centroidNums,1);
    y_fix = centroids(centroidNums,2);
    
    % calc warp from centroids
    tform = fitgeotrans([x_mov, y_mov], [x_fix, y_fix], 'Similarity');
    tform = maketform('affine', tform.T);
    
    % post-processing, calculate offset of centroid of interest from ideal position.  store this, and apply below.    
    modelCentroid_afterWarp = tformfwd(tform, modelCentroids(i,1), modelCentroids(i,2));
    delta = centroids(i,:) - modelCentroid_afterWarp;

    % NEXT, APPLY TO CENTROID OF INTEREST AND BACK-CALCULATE THE PARAMETERIZATION (p,q) ------------------->

    % calc model instance
    s = s0 + Sj*p_model(i);
    [A,b] = LK_qtoA(q_model(:,i), weights);
    s = A*s + repmat(b, [1, size(s,2)]);
    
    % register to get initialization
    [X, Y] = tformfwd(tform, s(1,:), s(2,:));
    s_init = [X; Y];
    % apply correction (from above)
    s_init = s_init + repmat(delta(:),[1, size(s_init,2)]);
    s_init_vec = reshape(permute(s_init,[2 1]),[prod(size(s_init)),1]);
    
    % calc corresponding q_init
    for j = 1:size(Sja,3),
        q_init(j,i) = dot(Sja_vec(:,j), (s_init_vec - s0_vec));
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