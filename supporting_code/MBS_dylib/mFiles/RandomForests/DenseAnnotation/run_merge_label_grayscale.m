%% run merge_label_grayscale

clear all

%
[grayscale_stack,sizeC,sizeZ,sizeT] = bf_openStack('grayscale_images.tif');
[label_stack,sizeC,sizeZ,sizeT] = bf_openStack('label_images.tif');

%%
for i = 1:size(label_stack,3),

    [merged(:,:,i)] = merge_label_grayscale(label_stack(:,:,i),grayscale_stack(:,:,i));
    
end

%% save

bfsave(merged, 'merged_images.tif');

%% unmerge
for i = 1:size(label_stack,3),

    [merged(:,:,i)] = merge_label_grayscale(label_stack(:,:,i),grayscale_stack(:,:,i));
    
    [label_stack2(:,:,i), grayscale_stack2(:,:,i)] = unmerge_label_grayscale(merged(:,:,i));
    
end

%% save

bfsave(grayscale_stack2, 'grayscale_images2.tif');
bfsave(label_stack2, 'label_images2.tif');
