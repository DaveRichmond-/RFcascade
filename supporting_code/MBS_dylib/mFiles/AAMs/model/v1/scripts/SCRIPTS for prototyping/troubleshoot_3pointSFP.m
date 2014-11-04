%% 

SFPimage = zeros(size(X,1), size(X,2), size(g,2));

for i = 1:size(g,2),
    
    tmp = zeros(size(X,1), size(X,2));
    tmp(CC.PixelIdxList{1,1}) = g(:,i);
    SFPimage(:,:,i) = tmp;
    
end

%%

fname = strcat('test SFP of boundary 2.avi');
writerObj = VideoWriter(fname);
open(writerObj);

fhandle = figure;

for i=1:size(SFPimage,3);
    
    image_variation = SFPimage(:,:,i);
    imagesc(image_variation),
%    colormap('gray'),
    axis tight
    set(gca,'XTick',[],'YTick',[]),
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
    
end

close(writerObj),

%% select subset of g vectors

n = 1;
m = size(g_store,2);

g = g_store(:,n:m);

%%

% make all image vectors zero-mean, and one-variance

g = g - repmat(mean(g,1), [size(g,1),1]);
g = g ./ repmat(sqrt(sum(g.^2,1)), [size(g,1),1]);

%%

[mean_image_vector, R, Psi, Lambda, PsiT] = myPCA(g);

