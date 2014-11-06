function [Offsets] = lineOffsets(centroids, numOffsets, offsetScale)

% make numOffsets odd
if (mod(numOffsets,2) == 0),
    numOffsets = numOffsets + 1;
end

Offsets = zeros(size(centroids,1), 2, numOffsets);

for i = 1:size(Offsets,1),
    if (i == 1),
        maxD = centroids(i+1,:) - centroids(i,:);
        maxU = -maxD;
    elseif (i == size(Offsets,1)),
        maxU = centroids(i-1,:) - centroids(i,:);
        maxD = -maxU;
    else
        maxU = centroids(i-1,:) - centroids(i,:);
        maxD = centroids(i+1,:) - centroids(i,:);
    end

    numEachDir = (numOffsets-1)/2;
    midIndx = (numOffsets+1)/2;
%     Offsets(i,:,midIndx) = 0;
    for j = 1:numEachDir,
        Offsets(i,:,midIndx - j) = (maxU/numEachDir)*j;
        Offsets(i,:,midIndx + j) = (maxD/numEachDir)*j;
    end
    
end

Offsets = offsetScale * Offsets;