%% test out working with appearance models

clear all

%% load data

% user params

image_fname = 'concat_f25-125_CROP_REG_halfFrames.tif';
data_fname = 'somite12.txt';

% load image stack

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack(image_fname);

% load dataset

[headers, table] = readTextFiles2(data_fname, 1, 6);

% add somite #
table(:,7) = 12;

% correct for Fiji zero indexing

table(:,[2:3]) = table(:,[2:3]) + ones(size(table,1),2);

%% build shape model from frames in data set

[xbar, R, Sigma, Psi, Lambda, V] = make_shape_model(table, 2);


%% find set of points in shape-normalized space lying on lattice with spacing dX = dY ~ 0.02 
%  and corresponding to (dilated) mask from mean somite shape

% 

numpoints = size(xbar,1)/2;

% create mask

[X Y] = meshgrid([-1 : 2/100 : 1]);
IN = inpolygon(X, Y, [xbar(1:numpoints); xbar(1)], [xbar(numpoints+1:2*numpoints); xbar(numpoints+1)]);
mask = reshape(IN, size(X));

% open and dilate mask

openRadius = 4;
dilRadius  = 5;

nhood = fspecial('disk', openRadius);
nhood = ceil(nhood);
mask = imopen(mask,nhood);

nhood = fspecial('disk', dilRadius);
nhood = ceil(nhood);
mask = imdilate(mask,nhood);

% flip mask to have same coordsys as vertex (shape) points

% mask = flipud(mask);

% plot mask

%{
figure,
imagesc(mask),
colormap('gray')
%}

%% store list of positions corresponding to pixels in mask

% find list of points in mask

% FIRST TRANSPOSE MASK, AND CORRESPONDING X,Y GRID TO ACHIEVE ROW-WISE RASTERING
%{
mask = mask';
X = X';
Y = Y';
%}

CC = bwconncomp(mask);

% plot mask from CC.PixelIdxList
%{
tempArray = zeros(size(X));
tempArray(CC.PixelIdxList{1,1}) = 1;
imagesc(tempArray);
%}

X = X(:);
Y = Y(:);
sampled_positions = [X(CC.PixelIdxList{1,1}) Y(CC.PixelIdxList{1,1})];

% transpose back
%{
mask = mask';
X = X';
Y = Y';
%}

% plot mask from sampled_positions

figure,
hold on

plot(sampled_positions(:,1), sampled_positions(:,2),'ok');
plot(xbar(1:numpoints), xbar(numpoints+1:2*numpoints),'ro','LineWidth',2)

%% calculate weights to transform single frame onto mean shape (eventually loop over all frames)

sigma = 0.1;
frame_num = 21;

indx = find(table(:,4) == frame_num);
shape_vector = table(indx, 2:3);

mean_vector = [xbar(1:numpoints), xbar(numpoints+1:2*numpoints)];

% first align shape_vector to mean_vector
%{
shape_vector = shape_vector - repmat(mean(shape_vector,1),[numpoints 1]);
shape_vector = shape_vector(:);
[shape_vector] = alignvectors(shape_vector, xbar);
shape_vector = [shape_vector(1:numpoints), shape_vector(numpoints+1:2*numpoints)];
%}

% test Matlab's thin plate spline

st = tpaps(mean_vector', shape_vector', 1);

warped_positions = fnval(st, sampled_positions')';

% test script with simple shape vectors
%{
% parameters for affine warp on mean_vector
b = 10;
s = 1;
theta = pi/4;

A = s*[cos(theta), -sin(theta); sin(theta), cos(theta)];

shape_vector = (A*mean_vector')' + b*ones(size(mean_vector,1), 2);
%}


% my implementation
%{
w = calc_weights_TPS(mean_vector, shape_vector, sigma); %shape_vector

% sample image according to warp model (weights from above)

[warped_positions] = TPS_warp(w, shape_vector, sampled_positions, sigma);

figure,
plot(warped_positions(:,1), warped_positions(:,2),'ob');
axis([-2 2 -2 2])
%}

%% 