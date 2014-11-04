function [hypotheses] = findCentroidsFromProbMap2(probMap, numHypotheses, params, plotFlag);

%{

find the multiple modes of a probability distribution.  designed for estimating the centroid of a segment from RF output

probMap: the probability map
numHypotheses: the number of modes per class to return

params is a structure containing the following parameters:
binSize: initial binning to find right region of the prob map to initialize mean-shift in
lambdaCoarse: window radius for initial mean-shift, to get average centroid
sigmaCoarse: bandwidth for Gaussian kernel in initial mean-shift
lambdaFine: window radius for subsequent mean-shifts, to get lots of local modes
sigmaFine: bandwidth...

%}

binSize = params.binSize;
lambdaCoarse = params.lambdaCoarse;
lambdaFine = params.lambdaFine;
sigmaCoarse = params.sigmaCoarse;
sigmaFine = params.sigmaFine;

% 1) FIND CENTROID OF HEAVILY SMOOTHED DISTRIBUTION 

% bin probability map ---------->

[M] = binImage(probMap, binSize);

% find bin with max cumulative prob, and Center of Mass within that bin, return position relative to whole image coords --------------->

[indxI, indxJ] = find(M == max(max(M)));

CoM = [indxI-1, indxJ-1]*binSize + centerOfMass(probMap(...
    (indxI-1)*binSize+1 : min(indxI*binSize,size(probMap,1)), ...
    (indxJ-1)*binSize+1 : min(indxJ*binSize,size(probMap,2))));

% run mean shift from this starting point --------------------->

mu(1,1) = CoM(2);
mu(1,2) = CoM(1);

for i = 2:100,
    mu(i,:) = mean_shift(probMap, mu(i-1,:), lambdaCoarse, sigmaCoarse);
    diff = pdist([mu(i,:); mu(i-1,:)]);
    if diff<2,
        break
    end
end

%

centroid = mu(end,:);

clear M p q m n indxI indxJ CoM mu

% 2) SAMPLE NEIGHBORHOOD AROUND FIRST CENTROID

L_bndry = max([floor(centroid(1) - binSize/4); 1]);
R_bndry = min([floor(centroid(1) + binSize/4) + 1; size(probMap,2)]);
T_bndry = max([floor(centroid(2) - binSize/4); 1]);
B_bndry = min([floor(centroid(2) + binSize/4) + 1; size(probMap,1)]);

[X Y] = meshgrid([L_bndry:R_bndry],[T_bndry:B_bndry]);
storeX = X;

X = X(:);
Y = Y(:);

for i = 1:length(X),
    pdf(i) = probMap(Y(i),X(i));        % check this!
end
pdf = pdf(:);
pdf = pdf / sum(pdf);

for i = 1:numHypotheses,
    
    indx = sampleProbDistrIndx(pdf);
    xy_init(i,:) = [X(indx), Y(indx)];
    
end

% 3) RUN MEAN-SHIFT ON SAMPLED POSITIONS

for j = 1:numHypotheses,
    
    mu = xy_init(j,:);
    
    for i = 2:100,
        mu(i,:) = mean_shift(probMap, mu(i-1,:), lambdaFine, sigmaFine);
        diff = pdist([mu(i,:); mu(i-1,:)]);
        if diff<2,
            break
        end
    end
    
    hypotheses(j,:) = mu(end,:);
    clear mu
    
end

% 4) visualize

if plotFlag,

    figure,
    imagesc(probMap)
    hold on,
    
    plot(xy_init(:,1), xy_init(:,2), 'r*','MarkerSize',20)
    plot(hypotheses(:,1), hypotheses(:,2),'w*','MarkerSize',20)
    axis([centroid(1,1)-binSize/4 centroid(1,1)+binSize/4 centroid(1,2)-binSize/4 centroid(1,2)+binSize/4])
    
%     plot(mu(:,1), mu(:,2),'ko','MarkerSize',20)
%     plot(mu(end,1), mu(end,2),'ko','MarkerSize',10)
%     axis([mu(end,1)-binSize/2 mu(end,1)+binSize/2 mu(end,2)-binSize/2 mu(end,2)+binSize/2])

end