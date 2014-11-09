% [mean_image, eigen_images] = mean_eigen_images(gbar, all_X{i}, pixelList{i}, Psi);
for i = 1:21,
    appearance_model_movie(modelSegmentsAAM(i).A0, modelSegmentsAAM(i).Ai, 0.001, 1, 2, strcat('somite#',num2str(i)));
end