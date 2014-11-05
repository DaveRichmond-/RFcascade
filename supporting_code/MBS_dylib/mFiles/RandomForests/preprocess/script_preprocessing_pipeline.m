% run through the major pre-processing steps, to go from a grayscale and label image to training data
% 1. register all images against the first
% 2. calculate features (in FIJI)
% 3. choose "important" features
% 4. manually delete all_features
% 5. create synthetic rotations
% 6. crop all images to standard size (w=800, h=900)

%% 1. register all images against the first

clear all

base = '/Users/richmond/Data/Somites/Processed/all+synthetic/processing';

fname_list_grayscale = getFileNames(strcat(base, '/unregistered/grayscale'), '.tif');
fname_list_labels    = getFileNames(strcat(base, '/unregistered/labels'), '.tif');

for i = 1:length(fname_list_grayscale),
    
    i,
    
    grayscale = imread(strcat(base, '/unregistered/grayscale/', fname_list_grayscale{i}));
    label     = imread(strcat(base, '/unregistered/labels/',    fname_list_labels{i}));

    % currently uses the first image as the reference
    if (i == 1)
        im_fix = grayscale;
        [mask_fix, temp] = maskEmbryo(im_fix, 1e3, 1);
        mask_fix = uint8(mask_fix) * 255;
    end
    
    % generate mask
    im_mov = grayscale;
    [mask_mov, temp] = maskEmbryo(im_mov, 1e3, 1);
    mask_mov = uint8(mask_mov) * 255;
    
    % set parameters for registration
    [optimizer, metric] = imregconfig('monomodal');
    optimizer.MaximumIterations = 300;
    
    %[mov_registered, R_reg] = imregister(im_mov, im_fix, 'rigid', optimizer, metric);
    tform = imregtform(mask_mov, mask_fix, 'rigid', optimizer, metric);
    imref_R = imref2d(size(im_fix));                        % for some reason this is essential!

    label_registered     = imwarp(label,     tform, 'OutputView', imref_R, 'Interp', 'nearest');
    grayscale_registered = imwarp(grayscale, tform, 'OutputView', imref_R, 'Interp', 'linear');
 
    % save
    bfsave(uint8(label_registered),     strcat(base, '/registered/labels/',    fname_list_labels{i}))
    bfsave(uint8(grayscale_registered), strcat(base, '/registered/grayscale/', fname_list_grayscale{i}))

end

clear all,

%% 2. run FIJI macro "computeFeatures.ijm" for computing feature stacks, then move to /processing/registered/features/all_features

%% 3. 

% USER PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% turn on/off the XY position features
XYgridFlag = 0;
% select a smaller subset of "easy to calculate" features
easyFeaturesFlag = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

base = '/Users/richmond/Data/Somites/Processed/all+synthetic/processing/registered/features/';

fname_list_features = getFileNames(strcat(base, 'all_features'), '.tif');

% list of variable indices, sorted by importance
load(strcat(base, 'variable_importance.mat'));
clear feature_imp feature_imp_sorted indx_sorted var_imp

% remove first two b/c no XY mesh in current feature stacks
indx = feat_keep(3:end);

% select subset of easy to calc features
if (easyFeaturesFlag)
    indx = indx([1:5,7,9:14,16:18,21,23:27]);
end

%
mkdir(strcat(base, 'impFeatures'));

for i = 1:length(fname_list_features),
    
    i,
    
    % load image
    im = bf_openStack(strcat(base, 'all_features/',fname_list_features{i}));

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
        im_comp = im(:,:,indx);
    end
    
    bfsave(single(im_comp), strcat(base, 'impFeatures/',fname_list_features{i}));
    clear im im_comp
    
end

% clear all,

%% 4. don't forget to manually delete all_features.

%% 5. generate synthetic training data, by rotating originals

clear all,

% USER PARAMETERS %%%%%%%%%%%%%%%%%%%
% max rotation angle.  rotations are within +/- max_angle.
max_angle = 10;
% # of random rotations
num_synth = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

origBase  = '/Users/richmond/Data/Somites/Processed/all+synthetic/processing/registered/';
synthBase = '/Users/richmond/Data/Somites/Processed/all+synthetic/final/';

fname_list_grayscale = getFileNames(strcat(origBase, 'grayscale'), '.tif');
fname_list_labels    = getFileNames(strcat(origBase, 'labels'), '.tif');
fname_list_features  = getFileNames(strcat(origBase, 'features/impFeatures'), '.tif');

% table of random rotations
rnd_rot = zeros(length(fname_list_grayscale),num_synth+1);
rnd_rot(:,2:end) = rand(length(fname_list_grayscale),num_synth)*2*max_angle - max_angle;

save(strcat(synthBase,'table_of_rotations.mat'),'rnd_rot');


for i = 1:length(fname_list_grayscale),
    
    i,
    
    grayscale = imread(strcat(origBase,       'grayscale/',             fname_list_grayscale{i}));
    label     = imread(strcat(origBase,       'labels/',                fname_list_labels{i}));
    features  = squeeze(bf_openStack(strcat(origBase, 'features/impFeatures/',  fname_list_features{i})));
        
    for j = 1:size(rnd_rot,2),
        
        % rotate
        grayscale_rot = imrotate(grayscale, rnd_rot(i,j), 'bilinear');
        label_rot     = imrotate(label,     rnd_rot(i,j), 'nearest');
        features_rot  = zeros(size(label_rot,1),size(label_rot,2),size(features,3));
        for k = 1:size(features,3),
            features_rot(:,:,k) = imrotate(features(:,:,k), rnd_rot(i,j),'bilinear');
        end
        
        % save
        bfsave(uint8(grayscale_rot), strcat(synthBase,'grayscale/',fname_list_grayscale{i}(1:strfind(fname_list_grayscale{i},'.tif')-1),'_',num2str(j),'.tif'))
        bfsave(uint8(label_rot),     strcat(synthBase,'labels/',   fname_list_labels{i}(1:strfind(fname_list_labels{i},'.tif')-1),'_',num2str(j),'.tif'))
        bfsave(single(features_rot), strcat(synthBase,'features/', fname_list_features{i}(1:strfind(fname_list_features{i},'.tif')-1),'_',num2str(j),'.tif'))
        
    end
    
end

% clear all

%% 6.  crop

% (i) firest copy topleftPos from .Numbers into workspace

% USER PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%
width  = 800;
height = 900;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

base = '/Users/richmond/Data/Somites/Processed/all+synthetic/final/';

fname_list_grayscale = getFileNames(strcat(base, 'grayscale'), '.tif');
fname_list_labels    = getFileNames(strcat(base, 'labels'), '.tif');
fname_list_features  = getFileNames(strcat(base, 'features'), '.tif');

cd(base);
mkdir('crop');

for i = 1:length(fname_list)
    
    % for 2d images
    %{
    im = imread(fname_list{i});
    im = im(topleftPos(i,2):topleftPos(i,2)+height-1, topleftPos(i,1):topleftPos(i,1)+width-1);
    %}

    % for stacks
    [im,sizeC,sizeZ,sizeT] = bf_openStack(fname_list{i});
    im = im(topleftPos(i,2):topleftPos(i,2)+height-1, topleftPos(i,1):topleftPos(i,1)+width-1, :);
    
    output_name = strcat('crop/',fname_list{i});
    bfsave(im, output_name);
    
end