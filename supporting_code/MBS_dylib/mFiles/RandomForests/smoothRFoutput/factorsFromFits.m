function [unaryFactors, pairwiseFactors, fitMasks] = factorsFromFits(segmentsFit, costs, lambdaU, lambdaPW, probMap, centroidStats)

% calculate weights of each fit (from cost and RF output) and return prob map

% useful constants
num_classes = size(probMap,3);
num_FGclasses = num_classes - 1;
numFits = size(segmentsFit,4);

% initialize
fitMasks        = false([size(probMap,1), size(probMap,2), num_FGclasses*numFits]);
[Xgrid, Ygrid]  = meshgrid([1:size(probMap,2)],[1:size(probMap,1)]);
unaryFactors    = zeros(num_FGclasses, numFits);
pairwiseFactors = zeros(numFits, numFits, num_FGclasses-1);

% reshape for easy indexing
probMapVec = reshape(probMap, [prod([size(probMap,1),size(probMap,2)]), num_classes]);

% unary Factors and fitMasks
for c = 1:num_FGclasses;
    
    for f = 1:numFits,
        
        %
        slice_indx = (c-1)*numFits + f;
        
        % create 0-1 mask for particular instance, and store indices of sparse 1's
        xv = segmentsFit(1,:,c,f)';
        yv = segmentsFit(2,:,c,f)';
        
        [IN, ON] = inpolygon(Xgrid, Ygrid, [xv; xv(1)], [yv; yv(1)]);
        fitMasks(:,:,slice_indx) = IN - ON;
        fgIndices = reshape(fitMasks(:,:,slice_indx), [prod([size(fitMasks,1), size(fitMasks,2)]), 1]);
        
        % calc unaries for each slice
        unaryFactors(c,f) = (sum(probMapVec(fgIndices,c+1))  / sum(fgIndices)) * exp(-lambdaU*costs(c,f));
        
    end
    
end

% pairwise Factors
for c = 1:num_FGclasses-1;
    
    for fL = 1:numFits,

        % calc centroid of fit (between points 4 and 8)
        centroidL = squeeze((segmentsFit(:,4,c,fL) + segmentsFit(:,8,c,fL)) / 2);

        for fR = 1:numFits,
            
            % calc centroid of fit (between points 4 and 8)
            centroidR = squeeze((segmentsFit(:,4,c+1,fR) + segmentsFit(:,8,c+1,fR)) / 2);
            
            % distance
            dist_centroids = norm(centroidL - centroidR); %sqrt(sum((centroidL - centroidR).^2,2));

            % calc prob of this distance under model
            pairwiseFactors(fL,fR,c) = normpdf(dist_centroids, centroidStats.mu(c), lambdaPW*sqrt(centroidStats.s2(c)));
            
        end
        
    end
    
end

% display
% display('The Unary Factors are:')
% unaryFactors
% 
% display('The Pairwise Factors are:')
% pairwiseFactors