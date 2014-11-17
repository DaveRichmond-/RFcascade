function [im] = appearance_model_figure(mean_image, eigen_images, p_axis, b_val)


imagesc(mean_image + eigen_images(:,:,p_axis)*b_val),
text(1,size(eigen_images,2),strcat('b(1) = ',num2str(b_val)),'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',24)
colormap('gray'),
axis tight
set(gca,'XTick',[],'YTick',[]),

im = mean_image + eigen_images(:,:,p_axis)*b_val;

%print('-dtiff',strcat('model_axis = ',num2str(p_axis),'_b = ',num2str(b_val),'.tif'))