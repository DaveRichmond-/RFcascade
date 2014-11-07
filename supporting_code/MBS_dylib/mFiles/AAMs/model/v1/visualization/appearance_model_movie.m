function appearance_model_movie(mean_image, eigen_images, Lambda, p_axis, n_sigma, varargin)

%

if nargin >= 6,
    fname = varargin{1};
else
    fname = '';
end

fname = strcat(fname,'_appearance variation along principle axis ',num2str(p_axis),'.avi');
writerObj = VideoWriter(fname);
open(writerObj);

lambda = Lambda(p_axis);
sigma_b = sqrt(lambda);

range_b = n_sigma*sigma_b;
deltaS = range_b / 9;
b_vals = [ [0 : -deltaS : -range_b], [-range_b : deltaS : range_b], [range_b : -deltaS : 0] ];

fhandle = figure;

for i=1:length(b_vals);
    
    image_variation = mean_image + eigen_images(:,:,p_axis)*b_vals(i);
    imagesc(image_variation),
    text(1,size(eigen_images,2),strcat('b(1) = ',num2str(b_vals(i))),'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',24)
    colormap('gray'),
    axis tight
    set(gca,'XTick',[],'YTick',[]),
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
    
end

close(writerObj),