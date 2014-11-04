%% generate simple images for testing RFs using VIGRA library

im_LHS = zeros(512,256);
im_RHS = zeros(512,256);

for j = 1:size(im_RHS,2),
    
    for i = 1:size(im_RHS,1),
        
        if mod(i+j,2),
            
            im_RHS(i,j) = 255;
            
        end
        
    end
    
end

% over-write im_RHS with ones
% im_RHS = 255*ones(512,256);

train_im = uint8([im_LHS, im_RHS]);
test_im = train_im';

figure(1)
imagesc(train_im);
colormap('gray')

figure(2)
imagesc(test_im);
colormap('gray')

% train_label image
train_label_im = [zeros(size(im_LHS)), ones(size(im_RHS))];

figure(3),
imagesc(train_label_im)

%% write images

imwrite(train_im,'train_image.tif');
imwrite(test_im,'test_image.tif');
imwrite(train_label_im,'train_label_image.tif');
