function [] = appearance_fit_movie_SCALED(som_num, mean_image, eigen_images, b_vals, b_axes, frames);

% with a black background (for slides)

% useful params

num_eigenImages = size(b_vals,1);

% open movie file

fname = strcat('fit som#',num2str(som_num),' to appearance model.avi');
writerObj = VideoWriter(fname);
open(writerObj);

fhandle = figure('color','black');

for i=1:length(frames);
    
    eigen_contr = zeros(size(eigen_images,1), size(eigen_images,2));
    
    for j = 1:length(b_axes),
        
        eigen_contr = eigen_contr + eigen_images(:,:,b_axes(j))*b_vals(b_axes(j),i);
        
    end
    
    image_variation(:,:,i) = mean_image + eigen_contr;
    
end

Cmin = min(min(min(image_variation)));
Cmax = max(max(max(image_variation)));

for i=1:length(frames),
    
    imagesc(image_variation(:,:,i), [Cmin Cmax]),
    %text(1,size(eigen_images,2),strcat('b(1) = ',num2str(b_vals(i))),'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',24)
    
    colormap('gray'),
    axis tight
    set(gca,'XTick',[],'YTick',[]),
    set(gca, 'Color', 'k'),
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
    
end

close(writerObj),