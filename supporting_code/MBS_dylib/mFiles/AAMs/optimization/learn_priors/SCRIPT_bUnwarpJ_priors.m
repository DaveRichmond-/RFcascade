%% 

%% read transformations

listing = dir(cd);
listing = listing(3:end);

%
for i = 1:length(listing),
    
    %
    tic,
    transf(i).name = listing(i).name,
    transf(i).c    = readbUnwarpJTransformsRAW(listing(i).name);
    toc,
    
end

%% test tracking

num_p = 1;
num_lambda = 1;
som_nums = [7];

% precompute
fname_model = 'som7_model.mat';
fname_image = 'f1-58_blur.tif';
last_frame = 58;

[shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,Ai,Ai_vec,Grad_A0_vec,Grad_Ai_vec,dWdp,dNdq,SFP_positions,pixelList,Trev_imageStack] = LK_precompute_Simultaneous(fname_model,fname_image,last_frame,num_p,num_lambda);

% initiate warp
load('fullDataSet.mat'),

som_indx = find(dataSet(:,7) == som_nums);
frame_indx = find(dataSet(som_indx,4) == last_frame);
s_pos = dataSet(som_indx(frame_indx), 2:3)';
%s_pos_vec = reshape(permute(s_pos,[2 1]),[prod(size(s_pos)),1]);

%%

load('TI_transf_matrices.mat');

for t = 1:size(Trev_imageStack,3),
    
    %
    if t==1,
        x = s_pos(1,:);
        y = s_pos(2,:);
    else
        for i = 1:size(s_pos,2),
            
            x(i) = bilinear_interp(transf(t-1).c(:,:,1), y_old(i), x_old(i))+1;
            y(i) = bilinear_interp(transf(t-1).c(:,:,2), y_old(i), x_old(i))+1;
            
        end
    end
    
    % display
    imagesc(Trev_imageStack(:,:,t)),
    colormap('gray'),
    axis([231 392 150 550]),
    set(gca,'XTick',[],'YTick',[]),

    hold on,
    
    plot(x,y,'wo')
    pause(1)
    
    x_old = x;
    y_old = y;
    
end