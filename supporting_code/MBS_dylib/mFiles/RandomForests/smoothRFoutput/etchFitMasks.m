function [etchedFitMasks] = etchFitMasks(FitMasks, uncertainty_radius, open_radius)

% etch fitMasks by the same method as during gt generation

% initialize
etchedFitMasks = zeros(size(FitMasks));

%
for i = 1 : size(FitMasks, 3),
    
    %
    etchedMask = FitMasks(:,:,i);
    
    % shrink mask
    nhood = fspecial('disk', uncertainty_radius);
    nhood = ceil(nhood);
    etchedMask = imerode(etchedMask,nhood);
    
    % round regions
    nhood = fspecial('disk', open_radius);
    nhood = ceil(nhood);
    etchedMask = imopen(etchedMask,nhood);

    % re-assign
    etchedFitMasks(:,:,i) = etchedMask;
    
end