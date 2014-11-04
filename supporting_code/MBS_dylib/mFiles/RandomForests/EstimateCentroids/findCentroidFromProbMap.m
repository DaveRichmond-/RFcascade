function [centroid] = findCentroidFromProbMap(probMap, binSize, lambda, sigma, plotFlag);

%{

find the mode of a probability distribution.  designed for finding the centroid of a segment from RF output

probMap: the probability map
binSize: initial binning to find right region of the prob map to initialize mean-shift in
lambda: window radius for mean-shift
sigma: bandwidth for Gaussian kernel in mean-shift

%}

% bin probability map ---------->

[M] = binImage(probMap, binSize);

% find bin with max cumulative prob, and Center of Mass within that bin --------------->

[indxI, indxJ] = find(M == max(max(M)));

CoM = [indxI-1, indxJ-1]*binSize + centerOfMass(probMap(...
    (indxI-1)*binSize+1 : min(indxI*binSize,size(probMap,1)), ...
    (indxJ-1)*binSize+1 : min(indxJ*binSize,size(probMap,2))));

% run mean shift from this starting point --------------------->

mu(1,1) = CoM(2);
mu(1,2) = CoM(1);

for i = 2:100,
    mu(i,:) = mean_shift(probMap, mu(i-1,:), lambda, sigma);
    diff = pdist([mu(i,:); mu(i-1,:)]);
    if diff<2,
        break
    end
end

%

centroid = mu(end,:);

%
if plotFlag,

    figure,
    imagesc(probMap)
    hold on,
    
    plot(mu(:,1), mu(:,2),'ko','MarkerSize',20)
    plot(mu(end,1), mu(end,2),'ko','MarkerSize',10)
%     axis([mu(end,1)-binSize/2 mu(end,1)+binSize/2 mu(end,2)-binSize/2 mu(end,2)+binSize/2])

end