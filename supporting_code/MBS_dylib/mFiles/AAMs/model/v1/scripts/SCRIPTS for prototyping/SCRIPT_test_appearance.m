%% test out working with appearance models

clear all

%{
notes:
this script assumes that the dataSet has already been corrected for zero-indexing in Fiji (by adding one to all positions values)
dataSet should also contain the somite number in column 7 of the data table
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% user parameters %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

image_fname = 'Gblur_r=2.tif';
data_fname = 'fullDataSet.mat';
som_nums = [6:15];

%% load data, and calculate a few useful parameters

% load image stack and data

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack(image_fname);
load(data_fname);

numpoints = size(unique(dataSet(:,1)), 1);

%% calculate once and for all the optimal pixel spacing of the shape free patch
%{
for i = 1:length(som_nums),
    
    indx = find(dataSet(:,7) == som_nums(i));
    frames = unique(dataSet(indx,4));
    
    for j = 1:length(frames),
        
        frames_indx = find(dataSet(indx,4) == frames(j));
        temp_shape_mat(:,:,frames(j)) = dataSet(indx(frames_indx), [2,3]);
        
    end
    
    shape_norm(i) = sqrt(mean(sum(var(temp_shape_mat,1),2)));
    
end
clear i indx frames frames_indx temp_shape_mat
%}

%% build shape normalized space

% calculate mean vector from data set using make_shape_model (not very efficient way to do this)

[xbar_vector, R, Psi, Lambda, PsiT] = make_shape_model(dataSet);
xbar_mat = [xbar_vector(1:numpoints), xbar_vector(numpoints+1:2*numpoints)];

% find pixels lying within mean shape

[X Y] = meshgrid([-1 : 0.01 : 1]);
mask = inpolygon(X, Y, [xbar_vector(1:numpoints); xbar_vector(1)], [xbar_vector(numpoints+1:2*numpoints); xbar_vector(numpoints+1)]);
mask = reshape(mask, size(X));

% open and dilate mask

openRadius = 8;
dilRadius  = 14;

nhood = fspecial('disk', openRadius);
nhood = ceil(nhood);
mask = imopen(mask,nhood);

nhood = fspecial('disk', dilRadius);
nhood = ceil(nhood);
mask = imdilate(mask,nhood);

clear nhood

%% store list of positions corresponding to pixels in mask

% find list of points in mask

CC = bwconncomp(mask);

sampled_positions = [X(CC.PixelIdxList{1,1}) Y(CC.PixelIdxList{1,1})];

% plot mask from sampled_positions
%{
figure,
hold on

plot(sampled_positions(:,1), sampled_positions(:,2),'ok');
plot(xbar_mat, xbar_mat,'ro','LineWidth',2)
%}

%% use Thin-Plate Spline to warp somite appearance onto Shape-Free-Patch

%{
% initialize storage variables
store_somite_num = zeros(1,length(unique(dataSet(    ))));
store_frame(k) = frame_nums(j);
store_warped_image(:,:,k) = warped_image;
store_image_vector(:,:,k) = warped_image_vector;
%}

% set up meshgrid over full image

[X_image, Y_image] = meshgrid([1:size(imageStack(:,:,1),2)], [1:size(imageStack(:,:,1),1)]);

% 

k = 1;

for i = 1:length(som_nums),
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    frame_nums = unique(dataSet(som_indx,4));
    
    for j = 1:length(frame_nums),
        
        % read shape_vector out of dataSet
        
        frame_indx = find(dataSet(som_indx,4) == frame_nums(j));
        shape_mat = dataSet(som_indx(frame_indx), 2:3);
        
        % calculate warp function using Matlab's thin plate spline
        
        st = tpaps(xbar_mat', shape_mat', 1);
        warped_positions = fnval(st, sampled_positions')';
        
        % map image at warped_positions back into warped_image
        
        raw_image = imageStack(:,:,frame_nums(j));
        warped_image = zeros(size(X,1), size(X,2));
        
        for m = 1:length(CC.PixelIdxList{1,1}),
            
            %{
                warped_image(CC.PixelIdxList{1,1}(m)) = interp2( X_image,Y_image,imageStack(:,:,frame_nums(j)), ...
                warped_positions(m,1), warped_positions(m,2) );
            %}            
            
            warped_image(CC.PixelIdxList{1,1}(m)) = bilinear_interp(raw_image, warped_positions(m,2),...
            warped_positions(m,1));
        
        end
        
        % create vectorized image for PCA
        
        warped_image_vector = warped_image(CC.PixelIdxList{1,1});
        warped_image_vector = warped_image_vector(:);
        
        % store everything
        
        store_somite_num(k) = som_nums(i);
        store_frame(k) = frame_nums(j);
        store_warped_image(:,:,k) = warped_image;
        store_image_vector(:,k) = warped_image_vector;
        
        % store in structure
        %{
        SFP(k).somite_num = som_nums(i);
        SFP(k).frame = frame_nums(j);
        SFP(k).image = warped_image;
        SFP(k).image_vector = warped_image_vector;
        %}
        
        k = k+1;
        
    end
    
end

% old version
%{
k=1;
for frame_num = 21:51,
    
    som_indx = find(table(:,4) == frame_num);
    shape_mat = table(som_indx, 2:3);
    
    % use Matlab's thin plate spline
    
    
    
    % read out image at warped_positions into shape-free patch
    
    raw_image = imageStack(:,:,frame_num);
    
    warped_image = zeros(size(meshgrid([-1 : 2/100 : 1]),1)*size(meshgrid([-1 : 2/100 : 1]),2), 1);
    
    for i = 1 : length(CC.PixelIdxList{1,1}),
        
        warped_image(CC.PixelIdxList{1,1}(i)) = bilinear_interp(raw_image, warped_positions(i,2),...
            warped_positions(i,1));
        
    end
    
    warped_image = reshape(warped_image, [size(meshgrid([-1 : 2/100 : 1]),1), size(meshgrid([-1 : 2/100 : 1]),2)]);
    
    store_warped_image(:,:,k) = warped_image;
    k = k+1;
    
end
%}

%% pre-process image vector

% transfer to g-vector, following notation from Cootes et al.

g = store_image_vector;

% translate all vectors to be zero mean                                             % NOT SURE IF THIS IS NECESSARY OR NOT !!!!!!!!!!!!!!!!!!!

%g = g - repmat(mean(g,1), [size(g,1),1]);

% choose 'middle' vector as reference and align all vectors to it

mean_indx = ceil(size(g,2)/2);

[g_aligned, gbar, D_meannotprojected] = normalize_image_vectors(g, mean_indx);


%% test convergence of above step
%{
diff_gbar = diff(store_gbar,1,2);
norm_diff_gbar = sqrt(sum(diff_gbar.^2,1));

figure,
plot(norm_diff_gbar),
%}

%%

% do PCA

[mean_image_vector, R, Psi, Lambda, PsiT] = myPCA(g_aligned);

%% create mean image, and first few eigen-images

mean_image = zeros(size(X,1), size(X,2));
mean_image(CC.PixelIdxList{1,1}) = mean_image_vector;

eigen_images = zeros(size(X,1), size(X,2), 9);

for i=1:size(Psi,2),

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

print -dtiff mean_image.tif

figure,
for i = 1:9,
    
    subplot(3,3,i),
    imagesc(eigen_images(:,:,i)),
    colormap('gray'),
    set(gca,'XTick',[],'YTick',[]),
    
end

print -dtiff first_9eigenimages.tif

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

print -dtiff var_lost.tif

%% project data onto first principal component

% user parameters for plotting

p_axis = 1;

% 

b = Psi'*R;

h = figure;
hold on,

k = 1;
for i = 1:length(som_nums),
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    frame_nums = unique(dataSet(som_indx,4));
    
    plot(frame_nums,b(p_axis,k:k+length(frame_nums)-1),'LineWidth',2),
    k = k+length(frame_nums);
    
end

axis([1 100 -1.1 1.1])

xl = xlabel('Time [frame #]'); set(xl,'fontsize',20),
yl = ylabel(strcat('Weight b_',num2str(p_axis))); set(yl,'fontsize',20),
tit = title('Description of somite appearance through time'); set(tit,'fontsize',24),

print('-dtiff',strcat('b1_somite',num2str(som_nums(1)),'-',num2str(som_nums(end)),'.tif'))

%% calculate projection error



%% create movie to visualize appearance changes associated with first principal component

% user parameters to make a movie of the shape variation

p_axis = 1;

%

fname = strcat('variation along principle axis ',num2str(p_axis),'.avi');
writerObj = VideoWriter(fname);
open(writerObj);

lambda = Lambda(p_axis);
sigma_b = sqrt(lambda);

range_b = 4*sigma_b;
deltaS = range_b / 99;
b_vals = [ [0 : -deltaS : -range_b], [-range_b : deltaS : range_b], [range_b : -deltaS : 0] ];

fhandle = figure;

for i=1:length(b_vals);
    
    image_variation = mean_image + eigen_images(:,:,p_axis)*b_vals(i);
    imagesc(image_variation),
    text(1,1,strcat('b(1) = ',num2str(b_vals(i))),'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',16)
    colormap('gray'),
    axis tight
    set(gca,'XTick',[],'YTick',[]),
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
    
end

close(writerObj),

%% create image of given b-value

p_axis = 1;
b_vals = -1;

imagesc(mean_image + eigen_images(:,:,p_axis)*b_vals),
colormap('gray'),
axis tight
set(gca,'XTick',[],'YTick',[]),

print('-dtiff',strcat('model_axis = ',num2str(p_axis),'_b = ',num2str(b_vals),'.tif'))

%% create movie of fit to data with mean + 1st FEW eigenvector(s)

p_axis = [1];

% 

%{
b = Psi'*R;
%}

b = appear(1).b;


fhandle = figure;
hold on,

k = 1;
for i = 1:length(som_nums),
    
    % set up movie file
    
    fname = strcat('fit somite #',num2str(som_nums(i)),' to principle axis #',num2str(p_axis(1)),'-',num2str(p_axis(end)),'.avi');
    writerObj = VideoWriter(fname);
    open(writerObj);

    % same corresponding frame nums
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    frame_nums = unique(dataSet(som_indx,4));
    
    for j = 1:length(frame_nums),
        
        image_variation = mean_image;
        
        for b_indx = 1:length(p_axis),
            
             image_variation = image_variation + eigen_images(:,:,p_axis(b_indx))*b(p_axis(b_indx),k+j-1);
             
        end
        
        subplot(1,2,1),
        imagesc(flipud(image_variation)),
        text(1,1,strcat('b(',num2str(p_axis),') = ',num2str(b(p_axis,k+j-1))),'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',16)
        colormap('gray'),
        axis tight
        set(gca,'XTick',[],'YTick',[]),
        
        image_variation = mean_image;
        
        for b_indx = 1:size(b,1),
            
             image_variation = image_variation + eigen_images(:,:,b_indx)*b(b_indx,k+j-1);
             
        end
        
        subplot(1,2,2)
        imagesc(flipud(image_variation)),
        colormap('gray'),
        axis tight
        set(gca,'XTick',[],'YTick',[]),
        
        frame = getframe(fhandle);
        writeVideo(writerObj,frame);
        
    end
    k = k+length(frame_nums);
    
    close(writerObj),
    
end

%% create movies of somite data (after alignment, but before PCA)

% 

b = Psi'*R;

fhandle = figure;
hold on,

k = 1;
for i = 1:length(som_nums),
    
    % set up movie file
    
    fname = strcat('fit somite #',num2str(som_nums(i)),' using all principle axes.avi');
    writerObj = VideoWriter(fname);
    open(writerObj);

    % same corresponding frame nums
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    frame_nums = unique(dataSet(som_indx,4));
    
    for j = 1:length(frame_nums),
        
        image_variation = mean_image;
        
        for b_indx = 1:size(b,1),
            
             image_variation = image_variation + eigen_images(:,:,b_indx)*b(b_indx,k+j-1);
             
        end
        
        imagesc(flipud(image_variation)),
        colormap('gray'),
        axis tight
        set(gca,'XTick',[],'YTick',[]),
        
        frame = getframe(fhandle);
        writeVideo(writerObj,frame);
        
    end
    k = k+length(frame_nums);
    
    close(writerObj),
    
end

%% plot parameters values from fit to first 2 principle axes

h = figure;
hold on,
plot(b(2,:),b(1,:),'LineWidth',2)
plot(b(2,:),b(1,:),'ro','LineWidth',2)

xl = xlabel('Weight b_2'); set(xl,'fontsize',20),
yl = ylabel('Weight b_1'); set(yl,'fontsize',20),
tit = title('Trajectory of two primary PCA weights'); set(tit,'fontsize',24),

print(h,'-dtiff',strcat('b2_b1.tif')),