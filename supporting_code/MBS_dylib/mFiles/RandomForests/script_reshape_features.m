listing = dir(pwd);
k=1;
m=1;

for i = 1:length(listing),
    if length(strfind(listing(i).name,'feature') >= 0)
        feat_list{k} = listing(i).name;
        k = k+1;
    end
    
    if length(strfind(listing(i).name,'label') >= 0)
        label_list{m} = listing(i).name;
        m = m+1;
    end
end
clear i k m listing

% set up array for all features
feat_array = NaN(1024*1024, 50);    % accounts for 4 images, with ~10% used pixels
count = 1;

for f_num = 1:length(feat_list),
    
    % load feature and label images
    feat_im = bf_openStack(feat_list{f_num});
    label_im = bf_openStack(label_list{f_num});
    
    % reshape
    feat_im = reshape(feat_im, [size(feat_im,1)*size(feat_im,2) size(feat_im,3)]);
    label_im = reshape(label_im, [size(label_im,1)*size(label_im,2) 1]);
    
    % down-sample background
    for i = 1:size(feat_im,1),
        if label_im(i) == 0,    %i.e. BG
            if rand < 0.02,
                label_array(count) = label_im(i);
                feat_array(count,:) = feat_im(i,:);
                count = count+1;
            end
        
        else
            label_array(count) = label_im(i);
            feat_array(count,:) = feat_im(i,:);
            count = count+1;
        end 
    end
                
    clear feat_im label_im
    
end

% crop feat and label array
feat_array = feat_array(1:count-1,:);
label_array = label_array(1:count-1);

%% save

% for saving
feat_array1 = feat_array(1:floor(size(feat_array,1)/2), :)';
feat_array2 = feat_array(floor(size(feat_array,1)/2)+1 : end, :)';

bfsave(feat_array1, 'feat_array1.tif');
bfsave(feat_array2, 'feat_array2.tif');
bfsave(label_array, 'label_array.tif');

%%

