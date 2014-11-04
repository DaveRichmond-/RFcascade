%% 

probMap = imread('level#2_image#7_probabilities21.tif');

binSize = 256;
lambda = 64;
sigma = 128;

% centroid = findCentroidFromProbMap(probMap, binSize, lambda, sigma, 1);

lambda2 = 32;
sigma2 = 16;
sample_sigma = 64;
numCentroids = 10;

% centroids = findCentroidsFromProbMap(probMap, binSize, lambda, lambda2, sigma, sigma2, sample_sigma, numCentroids, 1);

numModes = 10;
params.binSize = 256;
params.lambdaCoarse = 64;
params.sigmaCoarse = 128;
params.lambdaFine = 32;
params.sigmaFine = 16;
modes = findCentroidsFromProbMap2(probMap, numModes, params, 1);

% hold on,
% plot(modes(:,1), modes(:,2),'k*','MarkerSize',20)
% axis([centroid(1,1)-binSize/2 centroid(1,1)+binSize/2 centroid(1,2)-binSize/2 centroid(1,2)+binSize/2])

% original script (below) is now wrapped into above function
%{

%% bin prob map


M = probMap;
p = binSize;
q = binSize;

[m,n]=size(M); %M is the original matrix

M=sum( reshape(M,p,[]) ,1 );
M=reshape(M,m/p,[]).'; %Note transpose

M=sum( reshape(M,q,[]) ,1);
M=reshape(M,n/q,[]).'; %Note transpose

%% 
[indxI, indxJ] = find(M == max(max(M)));

CoM = [indxI-1, indxJ-1]*binSize + centerOfMass(probMap((indxI-1)*binSize+1 : indxI*binSize, (indxJ-1)*binSize+1 : indxJ*binSize));

hold on,
plot(CoM(2), CoM(1), 'w*')

%% 

lambda = 128;   % note: this is the radius of the window of points included in the average
sigma = 128;     % note: this is the bandwidth of the gaussian kernel

xy_init(1) = CoM(2);
xy_init(2) = CoM(1);

% mu(1,:) = xy_init;

mu = [440,160];

for i = 2:100,
    mu(i,:) = mean_shift(probMap, mu(i-1,:), lambda, sigma);
    diff = pdist([mu(i,:); mu(i-1,:)]);
    if diff<2,
        break
    end
end

%%

figure,
imagesc(probMap)
hold on,

plot(mu(:,1), mu(:,2),'wo','MarkerSize',10)
axis([320 480 40 240])

%}

%% repeat above, for all classes

% some user parameters
data_dir = '/Users/richmond/Analysis/SomiteTracker/RFs/real_data/Cascade_w_Smoothing/EstimateCentroids/3Levels_4ImagesPerLevel_20trees_Image7';
binSize = 256;
lambda = 64;
sigma = 128;

% get list of text files, store in fname (cell)

listing = dir(data_dir);
cd(data_dir);

k=1;

for i = 1:length(listing),
    
    if length(strfind(listing(i).name,'.tif') >= 0),

        fname_list{k} = listing(i).name;
        k = k+1;
        
    end
    
end
clear i k listing

% calc centroid for every image
centroidSet = [];
for i = 1:length(fname_list),
    
    probMap = imread(fname_list{i});
    centroidSet(i,:) = findCentroidFromProbMap(probMap, binSize, lambda, sigma, 1);
    
end

% first class (BG) has no meaningful centroid
centroidSet = centroidSet(2:end,:);

%%

[centroids] = findCentroidsFromProbMap(probMap, binSize, lambda1, lambda2, sigma1, sigma2, sample_sigma, numCentroids, plotFlag);