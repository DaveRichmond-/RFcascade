% [mean_image, eigen_images] = mean_eigen_images(gbar, all_X{i}, pixelList{i}, Psi);
for i = 1:21,
    appearance_model_movie(modelSegmentsAAM(i).A0, modelSegmentsAAM(i).Ai, 0.001, 1, 2, strcat('somite#',num2str(i)));
end
%%
i=20
im = appearance_model_figure(modelSegmentsAAM(i).A0, modelSegmentsAAM(i).Ai, 1, -1);
bfsave(single(im), 'app_20_n1.tif')