%% load var_importance list and apply to a directory of feature stacks

clear all

% set flag to 1 if you want to add XY-grid at this point, else 0
XYgridFlag = 1;

%
load('variable_importance.mat');
clear feature_imp feature_imp_sorted indx_sorted var_imp

% remove first two b/c no XY mesh in current feature stacks
indx = feat_keep(3:end);

% select a smaller subset of "easy to calculate" features
indx = indx([1:5,7,9:14,16:18,21,23:27]);

%
fname_list = getFileNames(pwd, '.tif');
mkdir(pwd,'/featSubset');

for i = 1:length(fname_list),
    
    i,
    
    % load image
    im = bf_openStack(fname_list{i});

    %
    if XYgridFlag,
        im_comp = single(zeros(size(im,1),size(im,2),length(indx)+2));
        im_comp(:,:,1:length(indx)) = im(:,:,indx);
        
        [X Y] = meshgrid([0:size(im,2)-1], [0:size(im,1)-1]);
        X = single(X);
        Y = single(Y);
        im_comp(:,:,length(indx)+1) = X;
        im_comp(:,:,length(indx)+2) = Y;
    else
        im_comp = single(zeros(size(im,1),size(im,2),length(indx)));
        im_comp(:,:,1:length(indx)) = im(:,:,indx);
    end
    
    bfsave(im_comp, strcat('featSubset/imp_',fname_list{i}));
    clear im
    
end