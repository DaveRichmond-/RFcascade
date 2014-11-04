fname = strcat('shape free evolution of somite 12.avi');
writerObj = VideoWriter(fname);
open(writerObj);

fhandle = figure;
colormap('gray');

for i = 1:31;
    
    imagesc(store_warped_image(:,:,i))
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
    
    clf,
    
end

close(writerObj),