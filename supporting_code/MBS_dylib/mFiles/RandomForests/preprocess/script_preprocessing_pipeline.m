% run through the major pre-processing steps, to go from a grayscale and label image to training data
% 1. register all images against the first
% 2. calculate features (in FIJI)
% 3. choose "important" features
% 4. manually delete all_features
% 5. create synthetic rotations
% 6. crop all images to standard size (w=800, h=900)

%% 0. generate label images - NOTE I HAVEN'T USED THIS IN THIS PIPELINE BEFORE.  PREVIOUSLY, I DID THIS STEP FIRST USING "SCRIPT_denseAnnotationFromGT_multiple.".  copied below...

% read in ground truth labeling of somite "parts" and create label image for RF training

%
position_nums = [1:9];
data_dir = pwd;

[dataSet, fname_list] = buildFullDataSet(data_dir, position_nums);
save('GT_labelling.mat','dataSet','position_nums','fname_list');

%% read data table into useful variable with position information associated with each somite

clear all,

fname = 'GT_labelling.mat';
load(fname);

num_somites = 21;

for emb_num = 1:length(position_nums),
    
    % re-assign points from data table to "pos variable"
    for i = 1:num_somites,
        
        if i < num_somites,
            indx = 5*(i-1) + [1:5,9:-1:7];
        else
            indx = 5*(i-1) + [1:5,8:-1:6];
        end
        
        % shift indx by corresponding position #
        indx = indx + 108*(emb_num-1);
        
        pos(:,:,i,emb_num) = dataSet(indx,[2:3])';                      % ERROR: NEED TO UPDATE POSITION TO RELEVANT EMBRYO (SHIFT BY 108 ROWS)
        
    end
    
end

%% create label image

% user parameters - CAREFUL, THESE SHOULD REMAIN CONSTANT FOR ALL TRAINING DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
uncertainty_radius = 2;
open_radius = 8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_image = getFileNames(pwd, '.tif');
train_image = imread(train_image{1});

figure(1),
imagesc(train_image);
colormap('gray');

for i = 1:size(pos,4),
    
    label_image = zeros(size(train_image));
    [X, Y] = meshgrid([1:size(train_image,1)]);
    
    for s = 1:num_somites;
        
        % create mask corresponding to the ith somite
        xv = pos(1,:,s,i)';
        yv = pos(2,:,s,i)';
        [IN, ON] = inpolygon(X, Y, [xv; xv(1)], [yv; yv(1)]);
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

%% 3. select out "important features"

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

%% 6. crop

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

%% NEXT STEPS ARE IN PREPARATION OF MODEL BUILDING!

%% 7. store all centroids from gt data, for "on the fly" backbone model building

clear all

% USER PARAMETERS %%%%%%%%%%%%%%%%%%%%%%
label_path  = '/Users/richmond/Data/gtSomites/only_originals/labels';
output_path = '/Users/richmond/Data/gtSomites/dataForModels';
num_classes = 22;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fname_list = getFileNames(label_path, '.tif');

for i = 1:length(fname_list),
    
    i,
    
    labelImage = imread(fname_list{i});

    binSize = floor(size(labelImage,1)/4);
    lambda =  floor(size(labelImage,1)/16);
    sigma =   floor(size(labelImage,1)/8);

    for c = 1:num_classes-1,
    
        mask = double(labelImage == c);
        modelCentroids(c,:,i) = findCentroidFromProbMap(mask, binSize, lambda, sigma, 0);
    
    end
    
    % visualize
    figure
    imagesc(labelImage)
    colormap('jet')
    hold on
    plot(modelCentroids(:,1,i),modelCentroids(:,2,i),'wo')
    
end

% save
save(strcat(output_path,'/rigidBackboneModel.mat'),'fname_list','modelCentroids')

clear all

%% 8. store all shapes from gt data

% USER PARAMTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_dir = '/Users/richmond/Data/gtSomites/dataForModels/buildAAMdataset/manual_full_labeling';
file_indx = 1:32;     % needs to be the same length as # of files
num_somites = 21;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get everything into one big table
[dataSet, fname_list] = buildFullDataSet(data_dir, file_indx);
clear data_dir

% load all points into better format for model building
pos = zeros(2,8,num_somites,length(fname_list));

for i = 1:length(fname_list),
    
    data = dataSet(dataSet(:,7)==file_indx(i), :);
    
    for s = 1:num_somites,
        
        if s < num_somites,
            indx = 5*(s-1) + [4:5,9:-1:7,1:3];
        else
            indx = 5*(s-1) + [4:5,8:-1:6,1:3];
        end
         
        pos(:,:,s,i) = data(indx,[2:3])';
        
    end
end

% flip LR

flip_indx = [3,4,6,8,9,10,12,13,14,15,18,19,20,22,23];

for i = 1:length(flip_indx)
    pos(1,:,:,flip_indx(i)) = 1024 - (pos(1,:,:,flip_indx(i)) - 1);
end

%% visualize

images_dir = '/Users/richmond/Data/gtSomites/dataForModels/buildAAMdataset/grayscale_images';
fname_list = getFileNames(images_dir, '.tif');

% visualize
for j = 1:32,
    figure,
    imagesc(imread(fname_list{j}))
    colormap('gray')
    hold on,
    for i = 1:21,
        plot(pos(1,:,i,j),pos(2,:,i,j),'ro','Markersize',10)
    end
end

%%

% change back to old naming scheme
xmat = pos;
clear pos

% initialize
xvec = zeros(size(xmat,1)*size(xmat,2), num_somites, length(fname_list));

% make zero mean, and vectorize
for i = 1:length(fname_list),
    
    for s = 1:num_somites,

        % center all shapes on origin
        xmat(:,:,s,i) = xmat(:,:,s,i) - repmat(mean(xmat(:,:,s,i),2), [1,size(xmat,2)]); 
        xvec(:,s,i) = reshape(permute(xmat(:,:,s,i),[2,1,3,4]), [size(xvec,1), 1, 1]); 
    end 
end

%%

% visualize
for i = 1:21,
    plot(xvec(1:8,i,11),-xvec(9:16,i,11))
    pause(0.5)
end

%% save

all_xvec = xvec;
fname_complete_list = fname_list;

save /Users/richmond/Data/gtSomites/dataForModels/buildAAMdataset/dataForShapeModel all_xvec fname_complete_list

%% 9. store all image patches from gt

clear all

% USER PARAMTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_dir = '/Users/richmond/Data/gtSomites/dataForModels/buildAAMdataset/grayscale_images';
data_dir = '/Users/richmond/Data/gtSomites/dataForModels/buildAAMdataset';
num_somites = 21;
numpoints = 8;
SFPpoints = 0:7;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load(strcat(data_dir,'/dataForShapeModel.mat'));
load(strcat(data_dir,'/allSomitePositions.mat'));
ShapeModel = buildShapeModelperSegment(data_dir, fname_complete_list);

% load all images
for i = 1:length(fname_complete_list),
    imageStack(:,:,i) = imread(strcat(image_dir,'/',fname_complete_list{i}));
end

for i = 1:num_somites,
    i,
    % unpack ShapeModel for ith somite
    xbar_vector = ShapeModel(i).xbar;
    xbar_mat = [xbar_vector(1:numpoints), xbar_vector(numpoints+1:2*numpoints)]';
    
    % build shape normalized space ------------------------------->
    
    % find pixels lying within mean shape
    [X, Y] = meshgrid([-1 : 0.01 : 1]);    
    mask = make_mask(X, Y, xbar_vector, SFPpoints, SFPpoints, 28, 2, 0.05);
    
    % find list of points in mask    
    CC = bwconncomp(mask);
    sampled_positions = [X(CC.PixelIdxList{1,1}) Y(CC.PixelIdxList{1,1})];
    
    % use Thin-Plate Spline to warp somite appearance onto Shape-Free-Patch ------------------------------->
    [X_image, Y_image] = meshgrid([1:size(imageStack(:,:,1),2)], [1:size(imageStack(:,:,1),1)]);
    
    for j = 1:length(fname_complete_list),
        
        % read shape_vector out of dataSet        
        shape_mat = pos(:,:,i,j);
        
        % calculate warp function using Matlab's thin plate spline        
        st = tpaps(xbar_mat, shape_mat, 1);
        warped_positions = fnval(st, sampled_positions')';
        
        % map image at warped_positions back into warped_image        
        raw_image = imageStack(:,:,j);
        warped_image = zeros(size(X,1), size(X,2));
        
        for m = 1:length(CC.PixelIdxList{1,1}),            
            warped_image(CC.PixelIdxList{1,1}(m)) = bilinear_interp(raw_image, warped_positions(m,2), warped_positions(m,1));            
        end
        
        % create vectorized image for PCA        
        warped_image_vector = warped_image(CC.PixelIdxList{1,1});
        warped_image_vector = warped_image_vector(:);
        
        % store everything        
        g(:,j) = warped_image_vector;
        
    end
    
    % translate all vectors to be zero mean
    g = g - repmat(mean(g,1), [size(g,1),1]);
    % INSTEAD OF ALIGNING ALL IMAGE VECTORS, SIMPLY make all image vectors one-norm !
    g = g ./ repmat(sqrt(sum(g.^2,1)), [size(g,1),1]);
    
    all_g{i} = g;
    pixelList{i} = CC.PixelIdxList{1,1};
    all_X{i} = X;
    all_Y{i} = Y;
    clear g CC X Y
    
end

%% visualize

for i = 4:4%length(fname_complete_list)
    [gbar, R, Psi, Lambda, ~] = myPCA(all_g{i});    
    
    [mean_image, eigen_images] = mean_eigen_images(gbar, all_X{i}, pixelList{i}, Psi);
    appearance_model_movie(mean_image, eigen_images, Lambda, 1, 2, strcat('somite#',num2str(i)));
end

%% save

save /Users/richmond/Data/gtSomites/dataForModels/buildAAMdataset/dataForAppModel3 all_g pixelList all_X all_Y fname_complete_list

%% 10. go back and generate centroids for non-rotated label images

clear all

fname_list_labels = getFileNames('/Users/richmond/Data/gtSomites/dataForModels/backup/unregistered_label_images', '.tif');

num_classes = 22;

for i = 1:length(fname_list_labels),
    
    i,
    
    labelImage = imread(fname_list_labels{i});

    binSize = floor(size(labelImage,1)/4);
    lambda =  floor(size(labelImage,1)/16);
    sigma =   floor(size(labelImage,1)/8);

    for c = 1:num_classes-1,
    
        mask = double(labelImage == c);
        modelCentroids(c,:,i) = findCentroidFromProbMap(mask, binSize, lambda, sigma, 0);

    end

end

%% visualize

fname_list = getFileNames('/Users/richmond/Data/gtSomites/dataForModels/backup/grayscale_images', '.tif');

for i = 1:length(fname_list_labels),
    
    i,
    figure
    grayImage = imread(fname_list{i});
    imagesc(grayImage);
    colormap('gray')
    hold on,
    plot(modelCentroids(:,1,i), modelCentroids(:,2,i),'ro')
    
end

%% save

all_centroids_unregistered = modelCentroids;
save

