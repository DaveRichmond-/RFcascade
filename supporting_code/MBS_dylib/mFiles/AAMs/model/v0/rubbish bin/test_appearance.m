%% test out working with appearance models

clear all

%% work with full training set?

% start by defining shape-normalized model space for appearance vector

load trainingSet.mat

% shift the training set by one pixel is X and Y, to account for Fiji zero indexing

trainingSet(:,[2:3]) = trainingSet(:,[2:3]) + ones(size(trainingSet,1),2);

% calculate the average shape xbar from the trainingSet

points = unique(trainingSet(:,1));
numpoints = size(points, 1);

[xbar, R, Sigma, Psi, Lambda, V] = make_shape_model(trainingSet, 2);

% plot average shape
%{
figure,
hold on,

plot(xbar(1:numpoints), xbar(numpoints+1:2*numpoints),'o',...
    'LineWidth',2,...
    'MarkerEdgeColor','k',...
    'MarkerSize',10),

fnplt(cscvn([[xbar(1:numpoints);xbar(1)]';[xbar(numpoints+1:2*numpoints);xbar(numpoints+1)]']),'r',2),
axis([-1 1 -1 1]),
%}

%% work with single somite data set!

%% find set of points in shape-normalized space lying on lattice with spacing dX = dY ~ 0.02 
%  and corresponding to (dilated) mask from mean somite shape

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

%% load data

% user params

image_fname = 'concat_f25-125_CROP_REG_halfFrames.tif';
data_fname = 'somite12.txt';

% load image stack

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack(image_fname);

% load dataset

[headers, table] = readTextFiles2(data_fname, 1, 6);

%% overlay data points on image

figure,
hold on,
colormap('gray'),

% set frame to plot

frame = 21;
indx = find(table(:,4) == frame);

% correct for offset in pixel vs. array values (Fiji starts at x = y = 0)

x_vals = table(indx,2) + ones(size(table(indx,2)));
y_vals = table(indx,3) + ones(size(table(indx,3)));

% plot

imagesc(imageStack(:,:,frame)),
plot(x_vals, y_vals, 'bo', 'LineWidth', 2),
fnplt(cscvn([[x_vals; x_vals(1)]'; [y_vals; y_vals(1)]']), 'r',1);

%% 