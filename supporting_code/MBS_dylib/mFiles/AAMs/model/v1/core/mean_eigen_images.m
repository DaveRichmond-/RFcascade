function [mean_image, eigen_images] = mean_eigen_images(mean_image_vector, X, pixelList, Psi);

mean_image = NaN(size(X,1), size(X,2));
mean_image(pixelList) = mean_image_vector;

eigen_images = NaN(size(X,1), size(X,2), size(Psi,2));

for i=1:size(Psi,2),

    temp_image = NaN(size(X,1), size(X,2));
    temp_image(pixelList) = Psi(:,i);
    eigen_images(:,:,i) = temp_image;

end
