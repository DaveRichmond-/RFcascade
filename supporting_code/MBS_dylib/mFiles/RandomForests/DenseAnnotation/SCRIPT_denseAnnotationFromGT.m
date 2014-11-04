%% script to read in ground truth labeling of somite "parts" and create label image for RF training

%%

position_nums = [1:24];
data_dir = pwd;

[dataSet, fname_list] = buildFullDataSet(data_dir, position_nums);
save('GT_labelling.mat','dataSet','position_nums','fname_list');

%% read data table into useful variable with position information associated with each somite

clear all,

fname = 'GT_labelling.mat';
load(fname);

num_somites = 24;

for emb_num = 1:length(position_nums),
    
    % shift i by total #
    
    % re-assign points from data table to "pos variable"
    for i = 1:num_somites,
        
        if i < num_somites,
            indx = 5*(i-1) + [1:5,9:-1:7];
        else
            indx = 5*(i-1) + [1:5,8:-1:6];
        end
        pos(:,:,i,emb_num) = dataSet(indx,[2:3])';                      % ERROR: NEED TO UPDATE POSITION TO RELEVANT EMBRYO (SHIFT BY 108 ROWS)
        
    end
    
end

%% create label image

% user parameters
uncertainty_radius = 2;
open_radius = 8;

train_image = imread('Focused 120807_bf_f0000_frame103.tif');

%{
figure(1),
imagesc(train_image);
colormap('gray');
%}

for i = 1:size(pos,4),
    
    label_image = zeros(size(train_image));
    [X, Y] = meshgrid([1:1024]);
    
    for s = 1:num_somites;
        
        % create mask corresponding to the ith somite
        xv = pos(1,:,s,i)';
        yv = pos(2,:,s,i)';
        [IN ON] = inpolygon(X, Y, [xv; xv(1)], [yv; yv(1)]);
        IN = IN - ON;       % only accept points IN the polygon, not on the boundary
        
        % shrink mask
        nhood = fspecial('disk', uncertainty_radius);
        nhood = ceil(nhood);
        IN = imerode(IN,nhood);
        
        % round regions
        nhood = fspecial('disk', open_radius);
        nhood = ceil(nhood);
        IN = imopen(IN,nhood);
        
        % turn mask into correct label
        IN = IN*s;
        label_image = label_image + IN;
        
    end
    
    % set the foreground
    FG = label_image > 0;
    
    % dilate FG to find the background
    nhood = fspecial('disk', 5*uncertainty_radius);
    nhood = ceil(nhood);
    BG = 1 - imdilate(FG,nhood);
    
    % replace "uncertain" regions with NaN values
    mask = zeros(size(label_image));
    mask = mask + BG + FG;
    mask(find(mask == 0)) = NaN;
    label_image = label_image .* mask;
    
    label_image = uint8(label_image);
    bfsave(label_image, strcat('label_image_',fname_list{i}(9:15),fname_list{i}(19:23),'.tif'));
    
    clear X Y xv yv IN ON nhood s FG BG mask label_image
    
end

%{

% display
figure(2),
imagesc(label_image),
colormap('jet')
set(gca,'XTick',[],'YTick',[]),

figure(3),
imagesc(mask),
colormap('gray'),
set(gca,'XTick',[],'YTick',[]),

%}

%% save

