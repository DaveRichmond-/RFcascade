clear all

% load GT centroids, and centroids from prob map
load('test_gtCentroid.mat')
load('image7_centroids.mat')

fname_list = getFileNames(pwd, '.tif');

for i = 1:length(fname_list)
    
    % read out xy coords
    x_mov = test_gtCentroid(:,1,i);
    y_mov = test_gtCentroid(:,2,i);
    x_fix = centroidSet(:,1);
    y_fix = centroidSet(:,2);
    
    % calc warp from centroids
    tform = fitgeotrans([x_mov, y_mov], [x_fix, y_fix], 'Similarity');
    
    % warp image
    im = imread(fname_list{i});
    imref_R = imref2d(size(im));
    im_registered = imwarp(im,tform, 'OutputView', imref_R, 'Interp', 'nearest');            % for some reason this is essential!
    
    % visualize    
    figure,
    imagesc(im_registered),
    
    % store
    im_store(:,:,i) = im_registered;
    
end

%% read in prob maps

fname_list = getFileNames(pwd, '.tif');

for i = 1:length(fname_list),
    im7_probMaps(:,:,i) = imread(fname_list{i});
end

%% turn registered centroids into prob map

class_list = [0:21];
weight = zeros(1,size(im_store,3));

for j = 1:size(im_store,3),

    for i = 1:length(class_list);
        
        c = class_list(i);
        % calculate weight for particular instance
        mask = im_store(:,:,j) == c;
        p = im7_probMaps(:,:,i);
        mask = mask(:);
        p = p(:);
        weight(j) = weight(j) + (p'*mask)/sum(mask);
        
    end

    weight(j) =  weight(j) / length(class_list);          % so that weights scale between [0,1]
    
end

%%

weightFlag = 1;

for i = 1:length(class_list);
    
    ClassMap = im_store == class_list(i);
    if weightFlag,
        for j = 1:size(ClassMap,3);
            ClassMap(:,:,j) = ClassMap(:,:,j)*weight(j);
        end
        smoothProbMap(:,:,i) = sum(ClassMap,3)/sum(weight,2);
    else
        smoothProbMap(:,:,i) = mean(ClassMap,3);
    end
    
    figure,
    imagesc(smoothProbMap(:,:,i))
    colormap('gray')

end
clear ClassMap

%% save

bfsave(scaleImage(smoothProbMap, '16bit', 1),'smoothProbMap_weighted_test.tif')