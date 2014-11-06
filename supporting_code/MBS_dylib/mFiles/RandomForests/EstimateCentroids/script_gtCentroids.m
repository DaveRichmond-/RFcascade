% comment out old approaches.  now use the approach of running meanShift.
%{
clear all,

fname = 'GT_labelling.mat';
load(fname);

num_somites = 21;

for emb_num = 1:length(position_nums),
    
    % re-assign points from data table to "pos variable"
    for i = 1:num_somites,
        
        if i < num_somites,
            indx = [5*(i-1)+3; 5*i+3];
        else
            indx = [5*(i-1)+3; 5*i+2];
        end
        
        % shift indx by corresponding position #
        indx = indx + 108*(emb_num-1);
        
        gtCentroid(i,1,emb_num) = mean(dataSet(indx,2));
        gtCentroid(i,2,emb_num) = mean(dataSet(indx,3));
        
    end
    
end

%%
save GT_Centroid.mat gtCentroid

%% clean gtCentroid: remove missing data point, flip many others L-R

load GT_Centroid.mat

keeper_indx  = [1:9,11:24];
flip_indx    = [0 0 1 1 0 1 0 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 1];

for i = 1:length(keeper_indx),

    clean_gtCentroid(:,:,i) = gtCentroid(:,:,keeper_indx(i));
    if flip_indx(i),
        clean_gtCentroid(:,1,i) = 1024 - clean_gtCentroid(:,1,i);
    end

end

%% test

fname_list = getFileNames(pwd, '.tif');
indx = [1:23];

for i = 1:length(fname_list)
    im = imread(fname_list{i});
    figure,
    imagesc(im),
    colormap('gray')
    hold on
    plot(clean_gtCentroid(:,1,indx(i)),clean_gtCentroid(:,2,indx(i)),'ro')
end

%% split into train and test

train_gtCentroid = clean_gtCentroid(:,:,[1:2:end]);
test_gtCentroid = clean_gtCentroid(:,:,[2:2:end]);

%% test split

load('test_gtCentroid.mat');

fname_list = getFileNames(pwd, '.tif');

for i = 1:length(fname_list)
    i
    im = imread(fname_list{i});
    figure,
    imagesc(im),
    colormap('gray')
    hold on
    plot(test_gtCentroid(:,1,i),test_gtCentroid(:,2,i),'ro')
end

%% 

%% JUST LOAD IN A SINGLE CENTROID MODEL AND SAVE AS .MAT

fname_list = getFileNames(pwd, '.txt');

[headers, table] = readTextFiles2(fname_list{1}, 1, 6);

% correct for zero-indexing of Fiji
modelCentroids = table(:,2:3) + ones(size(table,1),2);

save label_registered22_centroids.mat modelCentroids

%}

%% CALCULATE CENTROIDS AS COM FROM LABEL IMAGES.  SHOULD BE TRUER TO WHAT IS ESTIMATED FROM RF OUTPUT.

clear all

fname_list_labels = getFileNames('/Users/richmond/Data/gtSomites/only_originals/labels', '.tif');

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
