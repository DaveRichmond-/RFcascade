function [centroids] = findCentroidsFromProbMap(probMap, binSize, lambda1, lambda2, sigma1, sigma2, sample_sigma, numCentroids, plotFlag);

%{

find the multiple modes of a probability distribution.  designed for estimating the centroid of a segment from RF output

probMap: the probability map
binSize: initial binning to find right region of the prob map to initialize mean-shift in
lambda1: window radius for initial mean-shift, to get average centroid
sigma1: bandwidth for Gaussian kernel in initial mean-shift
lambda2: window radius for subsequent mean-shifts, to get lots of local modes
sigma2: bandwidth...
numCentroids: the number of modes to return
plotFlag: 1, for visualization

%}

% 1) FIND CENTROID OF HEAVILY SMOOTHED DISTRIBUTION 

% bin probability map ---------->

M = probMap;
p = binSize;
q = binSize;

[m,n] = size(M); %M is the original matrix

M = sum( reshape(M,p,[]) ,1 );
M = reshape(M,m/p,[]).'; %Note transpose

M = sum( reshape(M,q,[]) ,1);
M = reshape(M,n/q,[]).'; %Note transpose

% find bin with max cumulative prob, and Center of Mass within that bin --------------->

[indxI, indxJ] = find(M == max(max(M)));

CoM = [indxI-1, indxJ-1]*binSize + centerOfMass(probMap((indxI-1)*binSize+1 : indxI*binSize, (indxJ-1)*binSize+1 : indxJ*binSize));

% run mean shift from this starting point --------------------->

mu(1,1) = CoM(2);
mu(1,2) = CoM(1);

for i = 2:100,
    mu(i,:) = mean_shift(probMap, mu(i-1,:), lambda1, sigma1);
    diff = pdist([mu(i,:); mu(i-1,:)]);
    if diff<2,
        break
    end
end

%

centroid = mu(end,:);

clear M p q m n indxI indxJ CoM mu

% 2) SAMPLE NEIGHBORHOOD AROUND FIRST CENTROID

% sample_sigma = 64;      % hard-code this for now
xy_init = repmat(centroid, [numCentroids, 1]) + normrnd(0, sample_sigma, [numCentroids, 2]);

for j = 1:numCentroids,
    
    mu = xy_init(j,:)
    
    for i = 2:100,
        mu(i,:) = mean_shift(probMap, mu(i-1,:), lambda2, sigma2);
        diff = pdist([mu(i,:); mu(i-1,:)]);
        if diff<2,
            break
        end
    end
    
    centroids(j,:) = mu(end,:);
    
end

%3) VISUALIZE

if plotFlag,

    figure,
    imagesc(probMap)
    hold on,
    
    plot(centroids(:,1), centroids(:,2),'w*','MarkerSize',20)
    axis([centroid(1,1)-binSize/2 centroid(1,1)+binSize/2 centroid(1,2)-binSize/2 centroid(1,2)+binSize/2])
    
%     plot(mu(:,1), mu(:,2),'ko','MarkerSize',20)
%     plot(mu(end,1), mu(end,2),'ko','MarkerSize',10)
%     axis([mu(end,1)-binSize/2 mu(end,1)+binSize/2 mu(end,2)-binSize/2 mu(end,2)+binSize/2])

end