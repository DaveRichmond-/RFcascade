function [smoothProbMap] = probOfFit(probMap, segmentsFit, costs, lambda, backgroundBias)

% integrate probability map over a mask defined by AAM fit

%
num_classes = size(probMap,3);
class_list = [1:num_classes-1];
numFits = size(segmentsFit,4);

%
smoothProbMap = zeros(size(probMap));
smoothProbMap(:,:,1) = backgroundBias;

%
[Xgrid Ygrid] = meshgrid([1:1024],[1:1024]);
for c = 1:num_classes-1;
    
    for f = 1:numFits,
        
        % calculate weight for particular instance
        xv = segmentsFit(1,:,c,f)';
        yv = segmentsFit(2,:,c,f)';
        mask(:,:,f) = inpolygon(Xgrid, Ygrid, [xv; xv(1)], [yv; yv(1)]);
        
        weight(f) = (reshape(probMap(:,:,c+1),[1, prod([size(probMap,1),size(probMap,2)])]) * reshape(mask(:,:,f),[prod([size(mask,1),size(mask,2)]), 1]) / sum(sum(mask(:,:,f)))) * exp(-lambda*costs(c,f));     
        mask(:,:,f) = mask(:,:,f)*weight(f);
        
    end
    
    smoothProbMap(:,:,c+1) = sum(mask,3);   %/sum(weight(:));
    
end
    
smoothProbMap = smoothProbMap ./ repmat(sum(smoothProbMap,3),[1 1 num_classes]);