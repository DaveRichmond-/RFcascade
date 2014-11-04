%% scripted version of LK algorithm with appearance variation and global shape normalizing transform

%% Pre-compute steps

clear all

%% user defined dof in model

num_p = 1;
num_lambda = 1;

fname_model = 'som7_model.mat';
fname_image = 'grayscale_registered16_blur.tif';
% fname_image = 'f1-58_blur.tif';
last_frame = 1;

[shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,Ai,Ai_vec,Grad_A0_vec,Grad_Ai_vec,dWdp,dNdq,SFP_positions,pixelList,Trev_imageStack] = LK_precompute_Simultaneous(fname_model,fname_image,last_frame,num_p,num_lambda);

%% initiate warp

%
load('som20GT.mat'),
% load('fullDataSet.mat'),
som_nums = [20];

% select somite to track, and last frame (to start tracking from)


for i=1:length(som_nums),
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    frame_indx = find(dataSet(som_indx,4) == last_frame);
    s_pos = dataSet(som_indx(frame_indx), 2:3)';
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
    
    clear som_indx frame_inx s_pos A_temp b_temp s_unq
    
end

L_init = zeros(num_lambda,1);
L_init(1) = 0;

%%  shift initialization

p_init = 0;

q_init = q_init + [10 22 -45 -37]';

L_init(1) = 0;

%% Iterate steps

% user set parameters
reg_weights = 0*[1; 1; 1e-1; 1e-1; 1e3; 0];
step_size = ones(6,1);
num_iters = 500;
conv_thresh = [0.0005; 0.0005; 0.001; 0.001; 0.00005; 0.0002];

%
conv_flag_save = NaN(1,size(Trev_imageStack,3));

%
% figure,
for t = 1:1%size(Trev_imageStack,3),
    
    t,
    
    % step frame #
    raw_image = Trev_imageStack(:,:,t);
%{     
    if t~=1,
        q_init = q_final_save(:,t-1);
        p_init = p_final_save(:,t-1); 
    end
%}    
    [q_final_save(:,t), p_final_save(:,t), L_final(:,t), sp_final_save(:,:,t), q, p, L, sp_save, conv_flag_save(t)] = LK_GradDescent_Sim_wPriors(raw_image,A0_vec,Grad_A0_vec,Ai_vec,Grad_Ai_vec,dNdq,dWdp,...
        s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,SFP_positions,pixelList,q_init,p_init,L_init,conv_thresh,num_iters,reg_weights,step_size);
    
    if conv_flag_save(1,t) ~= 0,
        break,
    end
    
    p_init_save(:,t) = p_init;
    q_init_save(:,t) = q_init;
    
    % display
%{     
    imagesc(Trev_imageStack(:,:,t)),
    colormap('gray'),
    axis([231 392 250 650]),
    %axis([256 768 512 1024]),
    set(gca,'XTick',[],'YTick',[]),

    hold on,
    
    plot(sp_final_save(1,:,t),sp_final_save(2,:,t),'wo')
    pause(1)
%}     
    
end

%% movie of convergence in a single frame

% nice_image = imread('f58_noblur.tif');

t = 1;

save_flag = 0;
fname = 'temp.avi';%strcat('t=',num2str(t),'_c=1e-3*ones_ss=0p3.avi');
conv_time = size(p,2);
%{ 
if save_flag,
    writerObj = VideoWriter(fname);
    open(writerObj);
end
%} 
set(0,'DefaultFigureWindowStyle','normal'),
fhandle = figure;

for iter = 1:conv_time,
    clf,
    imagesc(Trev_imageStack(:,:,t)),
%     imagesc(nice_image);
    hold on,
    colormap('gray'),
    axis([550 750 700 900]),
    set(gca,'XTick',[],'YTick',[]),
    plot(sp_save(1,:,iter),sp_save(2,:,iter),'o',...
    'LineWidth',3,...
    'MarkerEdgeColor','w',...
    'MarkerSize',8),
    if save_flag,
        print('-dtiff',strcat('fit_frame=',num2str(iter))),
    else
        pause(0.25),
    end
end


%{
%% 
som_indx = 1;

shape_mean = shape(som_indx).xbar_vector;
shape_var = shape(som_indx).Psi(:,1);
shape_b_raw = p;

% shape instances

for i = 1:length(shape_b_raw),
    
    shape_inst(:,i) = shape_mean + shape_var * shape_b_raw(i);

end

%% test

shape_fit_movie(1, shape.xbar_vector, shape.Psi, shape_b_raw, [1:length(shape_b_raw)])

%% appearance data

%
som_indx = 1;

[appear_mean, appear_var] = mean_eigen_images(appear(som_indx).gbar, appear(som_indx).X, appear(som_indx).CC.PixelIdxList{1,1}, appear(som_indx).Psi);

% compress to first eigen-image

appear_var = appear_var(:,:,1);

% appearance instances

for i = 1:size(L,2),
    
    appear_inst(:,:,i) = appear_mean + appear_var * L(i);
    
end

%% warp SFP to account for shape change

SFP_warped = zeros(size(appear_inst));
[X_SFPw, Y_SFPw] = meshgrid([-1:0.01:1]);

sampled_positions = [X_SFPw(:), Y_SFPw(:)];

for k = 1:size(shape_inst,2),
    
    % 
    
    k
    
    % calculate warp function using Matlab's thin plate spline
    
    st = tpaps(reshape(shape_inst(:,k), [8 2])', reshape(shape_mean, [8 2])', 1);
    warped_positions = fnval(st, sampled_positions')';
    
    % translate to array indices
            
    SFPindx = warped_positions * 100 + 101;
    SFPindx_X = reshape(SFPindx(:,1), [size(X_SFPw,1), size(X_SFPw,2)]);
    SFPindx_Y = reshape(SFPindx(:,2), [size(Y_SFPw,1), size(Y_SFPw,2)]);
    
    % populate image
    
    for j = 1:size(X_SFPw,2),
        
        for i = 1:size(Y_SFPw,1),
            
            % interp
            
            if SFPindx_X(i,j) == 101 && SFPindx_Y(i,j) == 101,
                
                SFP_warped(i,j,k) = appear_inst(101,101,k);
                
            elseif (SFPindx_X(i,j) > 1) && (SFPindx_X(i,j) < size(appear_inst, 2)) && (SFPindx_Y(i,j) > 1) && (SFPindx_Y(i,j) < size(appear_inst, 1))
            
                SFP_warped(i,j,k) = bilinear_interp(appear_inst(:,:,k), SFPindx_Y(i,j), SFPindx_X(i,j));
            
            else
                
                SFP_warped(i,j,k) = NaN;
            
            end
            
        end
            
    end
    
end

%% movie of SFP_warped
som_num=1;
set(0,'DefaultFigureWindowStyle','normal'),
fname = strcat('somite #',num2str(som_num),' with shape.avi');
writerObj = VideoWriter(fname);
open(writerObj);

fhandle = figure('color','black');

for i = 1:length(frame_nums)
    
    imagesc(SFP_warped(:,:,i)),
    
    colormap('gray'),
    axis tight
    set(gca,'XTick',[],'YTick',[]),
    set(gca, 'Color', 'k'),
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
    
end

close(writerObj),

%% transform warped SFP to account for scaling / rotation

shape_data = sp_save;

% make domain for final image

xdata = squeeze(shape_data(1,:,:));
minX = min(min(xdata));
maxX = max(max(xdata));

ydata = squeeze(shape_data(2,:,:));
minY = min(min(ydata));
maxY = max(max(ydata));

widthX = ceil((maxX - minX)/2);
widthY = ceil((maxY - minY)/2);

[X Y] = meshgrid(1.5*[-widthX : widthX], 1.5*[-widthY : widthY]);

%

model_fit = zeros(size(Y,1), size(X,2), size(shape_inst,2));

for k = 1:size(shape_inst,2),
    
    k
    
    xmat = [sp_save(1,:,k); sp_save(2,:,k)]';
    xmat = xmat - repmat(mean(xmat,1),[8 1]);
    
    xvec = reshape(xmat, [16 1]);
    
    [x_aligned, T] = align_shape_vectors_wParams(xvec, shape_inst(:,k));
    
    % populate image
    
    for j = 1:size(X,2),
        
        for i = 1:size(Y,1),
            
            % calc corresponding positions in XY coords
            
            imagepos = [X(i,j), Y(i,j)];
            SFPpos = (T * imagepos')';
            
            % translate to array indices
            
            SFPindx = SFPpos * 100 + 101;
            
            % interp
            
            if SFPindx(1) == 101 && SFPindx(2) == 101,
                
                model_fit(i,j,k) = SFP_warped(101,101,k);
                
            elseif (SFPindx(1) > 1) && (SFPindx(1) < size(appear_inst, 2)) && (SFPindx(2) > 1) && (SFPindx(2) < size(appear_inst, 1))
            
                model_fit(i,j,k) = bilinear_interp(SFP_warped(:,:,k), SFPindx(2), SFPindx(1));
            
            else
                
                model_fit(i,j,k) = NaN;
            
            end
            
        end
            
    end
    
end

%% 

frame_nums = [1:size(p,2)];
fname = strcat('model fit with shape_scale_rot');
% writerObj = VideoWriter(fname, 'Uncompressed AVI');
% open(writerObj);

% fhandle = figure('color','black');

for i = 1:length(frame_nums)
    
    imagesc(model_fit(:,:,i)),
    
    colormap('gray'),
    axis tight
    set(gca,'XTick',[],'YTick',[]),
    set(gca, 'Color', 'k'),
    
    print('-dtiff',strcat(fname,'_frame=',num2str(i))),

%     frame = getframe(fhandle);
%     writeVideo(writerObj,frame);
    
end

% close(writerObj),
%}