%% test initialization and fittings

clear all
cd('/Users/richmond/Data/Somites/ModelForSmoothing')
% modelMap = imread('label_registered22.tif');
load ('label_registered22_centroids.mat');
load('modelSegmentsAAM.mat');

%%

% load test data
cd('/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing/MBS_AAMs/test')
probMap = bf_openStack('probs_stack.tif');
grayImage = imread('/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing/MBS_AAMs/test/grayscale_registered16.tif');



%%

[fishMask, LoG_Image] = maskEmbryo(grayImage, 1e3, 1);
threshMask = grayImage > 0;
mask = fishMask .* threshMask;

clear fishMask threshMask

%%

% pre-process the grayscale image for fitting ---------------------->
grayImage = double(grayImage);

% blur image
h = fspecial('gaussian',21,3);
grayImage = imfilter(grayImage, h, 'same');

% select foreground pixels for calculating stats
numer = sum(grayImage(:).*mask(:));
denom = sum(mask(:));

grayImage = grayImage - numer/denom;
numer = norm(grayImage(:).*mask(:));
grayImage = (grayImage / numer) * (denom / prod(size(modelSegmentsAAM.A0)));     % note scaling of norm to smaller patch

%%
%{
grayImage_vec = grayImage(:);

% translate imageStack vectors to be zero mean
grayImage_vec = grayImage_vec - mean(grayImage_vec);

% make raw image vector one-norm !!!!!!!!!!!!!!!!!!!!
grayImage_vec = grayImage_vec / norm(grayImage_vec); %sqrt(sum(grayImage_vec.^2));
grayImage = reshape(grayImage_vec, size(grayImage));
%}
%%

% other user set parameters
numCentroidsUsed = 11;
% numFits = 50;
numGDsteps = 500;
lambda = 10;
BG = 10;

[sp_save, q, p, L, cost] = fitAAMtoProbMap_gridSampleDebug(grayImage, probMap, modelCentroids, modelSegmentsAAM, numGDsteps, 2, 5);

%%

% [sp_save, q, p, L, cost] = fitAAMtoProbMap_ProjOut_gridSampleDebug(grayImage, probMap, modelCentroids, modelSegmentsAAM, numGDsteps, 1, 5);

%%

% visualize fit process
set(0,'DefaultFigureWindowStyle','normal'),
fhandle = figure('units','normalized','outerposition',[0 0 1 1])
conv_time = size(sp_save,3);

save_flag = 0;
fname = 'temp.avi';%strcat('t=',num2str(t),'_c=1e-3*ones_ss=0p3.avi');
if save_flag,
    writerObj = VideoWriter(fname);
    open(writerObj);
    set(0,'DefaultFigureWindowStyle','normal'),

end

for iter = 1:5:conv_time,
    %iter,
    
    clf,
    subplot(2,4,1),
    imagesc(grayImage),
    axis([150 450 100 450])
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
    subplot(2,4,8)
    plot(cost(1,1:iter-2)),
    %
    if save_flag,
        frame = getframe(fhandle);
        writeVideo(writerObj,frame);
    else
        pause(0.001),
    end
end
if save_flag,
    close(writerObj),
    set(0,'DefaultFigureWindowStyle','docked'),
end

set(0,'DefaultFigureWindowStyle','docked'),
