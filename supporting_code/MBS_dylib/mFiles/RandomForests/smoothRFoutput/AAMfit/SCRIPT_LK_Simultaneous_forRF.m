%% scripted version of LK algorithm with appearance variation and global shape normalizing transform

clear all

%% Pre-compute steps

load('/Users/richmond/Data/Somites/ModelForSmoothing/modelSegmentsAAM.mat')

% unpack
A0_vec = modelSegmentsAAM.A0_vec;
Grad_A0_vec = modelSegmentsAAM.Grad_A0_vec;
Ai_vec = modelSegmentsAAM.Ai_vec;
Grad_Ai_vec = modelSegmentsAAM.Grad_Ai_vec;
dNdq = modelSegmentsAAM.dNdq;
dWdp = modelSegmentsAAM.dWdp;
s0 = modelSegmentsAAM.s0;
s0_vec = modelSegmentsAAM.s0_vec;
Sj = modelSegmentsAAM.Sj;
Sj_vec = modelSegmentsAAM.Sj_vec;
Sja = modelSegmentsAAM.Sja;
Sja_vec = modelSegmentsAAM.Sja_vec;
weights = modelSegmentsAAM.weights;
SFP_positions = modelSegmentsAAM.SFP_positions;
pixelList = modelSegmentsAAM.pixelList;

%%

grayImage = imread('/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing/MBS_AAMs/test/grayscale_registered16_blur.tif');

% pre-process the grayscale image for fitting
grayImage = double(grayImage);
grayImage_vec = grayImage(:);

% translate imageStack vectors to be zero mean
grayImage_vec = grayImage_vec - mean(grayImage_vec);

% make raw image vector one-variance !!!!!!!!!!!!!!!!!!!!
grayImage_vec = grayImage_vec ./ sum(grayImage_vec.^2);               % old code actually used one-standard-deviation
grayImage = reshape(grayImage_vec, size(grayImage));

%% initiate warp

% select somite to track, and last frame (to start tracking from)
som_nums = [20];


%
load(strcat('som',num2str(som_nums),'GT.mat')),

num_p = 1;
num_lambda = 1;
%


for i=1:length(som_nums),
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    s_pos = dataSet(som_indx, 2:3)';
    s_pos_vec = reshape(permute(s_pos,[2 1]),[prod(size(s_pos)),1]);
    
    for j = 1:size(Sja,3),
        q_init(j,i) = dot(Sja_vec(:,j), (s_pos_vec - s0_vec));
    end

    [A_temp,b_temp] = LK_qtoA(q_init(:,i), weights);
    
    s_unq = (A_temp^-1)*(s_pos - repmat(b_temp,[1,size(s0,2)]));
    s_unq_vec = reshape(permute(s_unq,[2 1]),[prod(size(s_unq)) 1]);
    
    % back-calculate pp's
    
    for j = 1:size(Sj,3),
        
        p_init(j,i) = dot(Sj_vec(:,j), (s_unq_vec - s0_vec));
        
    end
end

L_init = zeros(num_lambda,1);
L_init(1) = 0;

%% Iterate steps

% visualize initialization
figure,
imagesc(grayImage),
colormap('gray'),
set(gca,'XTick',[],'YTick',[]),
hold on,

s = s0 + Sj*p_init(1);
[A,b] = LK_qtoA(q_init(:,1), weights);
s = A*s + repmat(b, [1, size(s,2)]);

plot(s(1,:),s(2,:))

% fit AAM instances to grayscale image ------------------------->

LKparams.reg_weights = 1e-3*[1; 1; 1e-1; 1e-1; 1e3; 0];
LKparams.step_size = 0.5*ones(6,1);
LKparams.num_iters = 10;
LKparams.conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005; 0.0002];

% visualize fit
figure,
imagesc(grayImage),
colormap('gray'),
set(gca,'XTick',[],'YTick',[]),
hold on,

for i = 1:1,
    
    i,
    [q_final_save(:,i), p_final_save(:,i), L_final(:,i), sp_final_save(:,:,i), q, p, L, sp_save, conv_flag_save(i), cost(i)] = LK_GradDescent_Sim_wPriors_forRF(grayImage, modelSegmentsAAM, q_init(:,i), p_init(:,i), L_init(:,i), LKparams);
        
    % display
    plot(sp_final_save(1,:,i),sp_final_save(2,:,i),'wo')
    
end

% visualize fit process
figure,
save_flag = 0;
fname = 'temp.avi';%strcat('t=',num2str(t),'_c=1e-3*ones_ss=0p3.avi');
conv_time = size(p,2);

if save_flag,
    writerObj = VideoWriter(fname);
    open(writerObj);
    set(0,'DefaultFigureWindowStyle','normal'),

end

fhandle = figure;

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