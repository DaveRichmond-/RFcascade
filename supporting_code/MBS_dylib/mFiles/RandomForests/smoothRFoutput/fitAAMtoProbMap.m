function [segmentsFit, costs] = fitAAMtoProbMap(grayImage, probMap, modelCentroids, modelSegmentsAAM, numCentroidsUsed, numGDsteps);

% some user defined parameters.  these are hard-coded for now, because they work pretty well.
% num_modes = 1;
params.binSize = floor(size(probMap,1)/4);
params.lambdaCoarse = floor(size(probMap,1)/16);
params.sigmaCoarse = floor(size(probMap,1)/8);
params.lambdaFine = floor(size(probMap,1)/32);
params.sigmaFine = floor(size(probMap,1)/64);

% useful variables
num_classes = size(probMap,3);

% 1) calc centroids for every class
for i = 2:num_classes,
    centroids(i,:) = findCentroidsFromProbMap2(probMap(:,:,i), 1, params, 0);
end
% first class (BG) has no meaningful centroid
centroids = centroids(2:end,:);

% initialize AAM instances ------------------->

[p_init, q_init] = initializeModelSegmentsAAM(centroids, modelCentroids, numCentroidsUsed, modelSegmentsAAM);
L_init = zeros(1,size(p_init,2));

% visualize initialization
%{
test_num = 2;
figure,
imagesc(grayImage),
colormap('gray'),
%axis([256 768 512 1024]),
set(gca,'XTick',[],'YTick',[]),
hold on,

s = modelSegmentsAAM.s0 + modelSegmentsAAM.Sj*p_init(test_num);
[A,b] = LK_qtoA(q_init(:,test_num), modelSegmentsAAM.weights);
s = A*s + repmat(b, [1, size(s,2)]);

plot(s(1,:),s(2,:))
pause(1)
%}

% fit AAM instances to grayscale image ------------------------->

LKparams.reg_weights = 1e-3*[1; 1; 1e-1; 1e-1; 1e3; 0];
LKparams.step_size = 0.5*ones(6,1);
LKparams.num_iters = numGDsteps;
LKparams.conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005; 0.0002];

for i = 1:size(probMap,3)-1,
    
    [segmentsFit(:,:,i), costs(i)] = LK_GradDescent_Sim_wPriors_forRF(grayImage, modelSegmentsAAM, q_init(:,i), p_init(:,i), L_init(:,i), LKparams);
    
%     [q_final_save(:,i), p_final_save(:,i), L_final(:,i), sp_final_save(:,:,i), q, p, L, sp_save, conv_flag_save(i), costs(i)] = LK_GradDescent_Sim_wPriors_forRFdebug(grayImage, modelSegmentsAAM, q_init(:,i), p_init(:,i), L_init(:,i), LKparams);
    

    % visualize fit
%{
    figure,
    imagesc(grayImage),
    colormap('gray'),
    set(gca,'XTick',[],'YTick',[]),
    hold on,
    plot(segmentsFit(1,:,i),segmentsFit(2,:,i),'wo')
    pause(1)
%}  
end

%{
% visualize fit process
fhandle = figure('units','normalized','outerposition',[0 0 1 1])
save_flag = 0;
fname = 'temp.avi';%strcat('t=',num2str(t),'_c=1e-3*ones_ss=0p3.avi');
conv_time = size(p,2);

if save_flag,
    writerObj = VideoWriter(fname);
    open(writerObj);
    set(0,'DefaultFigureWindowStyle','normal'),

end

for iter = 1:conv_time,
    %iter,
    
    clf,
    subplot(2,4,1),
    imagesc(grayImage),
    colormap('gray'),
    set(gca,'XTick',[],'YTick',[]),

    hold on,

    plot(sp_save(1,:,iter),sp_save(2,:,iter),'wo')

    subplot(2,4,2)
    plot(q(3,1:iter)),
    
    subplot(2,4,3)
    plot(q(4,1:iter)),
    
    subplot(2,4,4)
    plot(q(1,1:iter)),
    
    subplot(2,4,5)
    plot(q(2,1:iter)),
    
    subplot(2,4,6)
    plot(p(1,1:iter)),
    
    subplot(2,4,7)
    plot(L(1,1:iter)),
    
    % 
    
    %
    if save_flag,
        frame = getframe(fhandle);
        writeVideo(writerObj,frame);
    else
        pause(0.05),
    end
end

if save_flag,
    close(writerObj),
    set(0,'DefaultFigureWindowStyle','docked'),
end
%}