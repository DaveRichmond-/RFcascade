function [smoothProbMap] = probMapFromFits(probMap, segmentsFit, costs, lambda)

% calculate weights of each fit (from cost and RF output) and return prob map

% useful nums
num_classes = size(probMap,3);
class_list = [1:num_classes-1];
numFits = size(segmentsFit,4);

% initialize
smoothProbMap = single(zeros(prod([size(probMap,1), size(probMap,2)]), size(probMap, 3)));
thresh_mask   = false(prod([size(probMap,1), size(probMap,2)]), (num_classes-1)*numFits);
weight_mask   = single(zeros(prod([size(probMap,1), size(probMap,2)]), (num_classes-1)*numFits));
weight        = zeros(1, (num_classes-1)*numFits);

[Xgrid, Ygrid] = meshgrid([1:size(probMap,2)],[1:size(probMap,1)]);

probMap = reshape(probMap, [prod([size(probMap,1),size(probMap,2)]), num_classes]);

%
for c = 1:num_classes-1;
    
    for f = 1:numFits,
        
        % put everything into one big stack, for easy summing operation
        slice_indx = (c-1)*numFits + f;
        
        % create 0-1 mask for particular instance
        xv = segmentsFit(1,:,c,f)';
        yv = segmentsFit(2,:,c,f)';
        temp_mask = inpolygon(Xgrid, Ygrid, [xv; xv(1)], [yv; yv(1)]);
        thresh_mask(:,slice_indx) = reshape(temp_mask, [size(probMap,1), 1]);
        
        % calc weights for each slice, and create weight_mask
%         weight(slice_indx) = (probMap(:,c+1)' * thresh_mask(:,slice_indx) / sum(thresh_mask(:,slice_indx))) * exp(-lambda*costs(c,f));
%         weight_mask(:,slice_indx) = thresh_mask(:,slice_indx) * weight(slice_indx);
        weight(slice_indx) = (sum(probMap(thresh_mask(:,slice_indx),c+1))  / sum(thresh_mask(:,slice_indx))) * exp(-lambda*costs(c,f));
        weight_mask(thresh_mask(:,slice_indx),slice_indx) =  weight(slice_indx);

        % calc centroid of fit (between points 4 and 8
%         centroid(c,f,1) = (segmentsFit(1,4,c,f) + segmentsFit(1,8,c,f)) / 2;
%         centroid(c,f,2) = (segmentsFit(2,4,c,f) + segmentsFit(2,8,c,f)) / 2;
        
    end
    
end

% compute probabilities from thresh_mask and weight_mask ------------------------------->

% pre-calculate the constants
n = sum(thresh_mask, 2);
N = numFits;
p_class = zeros(size(probMap,1), 1);
sum_all_class_weights = sum(weight_mask,2);

% three cases
indx0 = n == 0;
indx1 = logical((n ~= 0) .* (n < N));
indx2 = n >= N;

% background class
p_class(indx0) = 1;
p_class(indx1) = 1 - n(indx1)/N;
p_class(indx2) = 0;
smoothProbMap(:,1) = p_class; %reshape(bckgrnd, [size(probMap,1), size(probMap,2)]);

%
for c = 1:num_classes-1;
    sum_class_weights = sum(weight_mask(:,(c-1)*numFits+1:c*numFits), 2);
    %
    p_class(indx0) = 0;
    p_class(indx1) = (n(indx1)/N) .* (sum_class_weights(indx1) ./ sum_all_class_weights(indx1));
    p_class(indx2) = sum_class_weights(indx2) ./ sum_all_class_weights(indx2);
    %
    smoothProbMap(:,c+1) = p_class;
end

% reshape
smoothProbMap = reshape(smoothProbMap, [size(Xgrid,1), size(Xgrid,2), num_classes]);