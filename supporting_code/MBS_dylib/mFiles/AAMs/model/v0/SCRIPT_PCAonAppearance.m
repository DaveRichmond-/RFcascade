%%

% pre-process image vector

normed_image_vector = store_image_vector;

% do PCA

[mean_image_vector, Psi, Lambda, PsiT] = myPCA(normed_image_vector);

%% create mean image, and first few eigen-images

mean_image = zeros(size(X,1), size(X,2));
mean_image(CC.PixelIdxList{1,1}) = mean_image_vector;

eigen_images = zeros(size(X,1), size(X,2), 9);

for i=1:9,

    temp_image = zeros(size(X,1), size(X,2));
    temp_image(CC.PixelIdxList{1,1}) = Psi(:,i);
    eigen_images(:,:,i) = temp_image;

end

% test mean from PCA
%{
test_flag = sum(sum(mean_image - mean(store_warped_image,3)));
if test_flag ~= 0,
    error('PCA returns incorrect mean')
end
%}

% test normalization of eigenimages
%{
for i = 1:9,
    
    norm(Psi(:,i))
    
end
%}

%% plot

figure,
imagesc(mean_image),
colormap('gray'),

figure,
for i = 1:size(eigen_images,3),
    
    subplot(3,3,i), % ceil(i/3), rem((i-1),3)+1)
    imagesc(eigen_images(:,:,i)),
    colormap('gray'),
    
end

%% evaluate variance associated with each principal component



var_lost = 1 - cumsum(Lambda)/sum(Lambda);

figure,
plot(var_lost,'r-','LineWidth',2),
xl = xlabel('number of PCA modes kept');
yl = ylabel('fraction of total variation LOST');
tit = title('Performance of PCA decomposition');
set(xl,'fontsize',20),
set(yl,'fontsize',20),
set(tit,'fontsize',24),