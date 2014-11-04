%% SCRIPT calculate non-local features from feature stack

%% open feature stack

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack('predict_probabilities.tif');

feature_stack = squeeze(imageStack);
clear imageStack size*

%% consider additing position XY features

%% calc features offset by 100 pixels, for 8 neighborhood

% user variables
offset = 200;

% create padded feature stack to sample from
pad_feature_stack = padarray(feature_stack,[offset offset 0],'replicate');

% initiate
TL = zeros(size(feature_stack));
TM = zeros(size(feature_stack));
TR = zeros(size(feature_stack));
ML = zeros(size(feature_stack));
MM = zeros(size(feature_stack));
MR = zeros(size(feature_stack));
BL = zeros(size(feature_stack));
BM = zeros(size(feature_stack));
BR = zeros(size(feature_stack));

%{
% fill TL (top left) feature stack
TL(1:offset, 1:offset, :) = repmat(feature_stack(1,1,:),[offset, offset, 1]);
TL(1:offset, offset+1:end, :) = repmat(feature_stack(1, 1:end-offset, :),[offset, 1, 1]);
TL(offset+1:end, 1:offset, :) = repmat(feature_stack(1:end-offset, 1, :),[1, offset, 1]);
TL(offset+1:end, offset+1:end, :) = feature_stack(1:end-offset, 1:end-offset, :);
%}

% fill top row
TL = pad_feature_stack(1:end-2*offset, 1:end-2*offset, :);
TM = pad_feature_stack(1:end-2*offset, offset+1:end-offset, :);
TR = pad_feature_stack(1:end-2*offset, 2*offset+1:end, :);

% fill middle row
ML = pad_feature_stack(offset+1:end-offset, 1:end-2*offset, :);
MM = pad_feature_stack(offset+1:end-offset, offset+1:end-offset, :);
MR = pad_feature_stack(offset+1:end-offset, 2*offset+1:end, :);

% fill middle row
BL = pad_feature_stack(2*offset+1:end, 1:end-2*offset, :);
BM = pad_feature_stack(2*offset+1:end, offset+1:end-offset, :);
BR = pad_feature_stack(2*offset+1:end, 2*offset+1:end, :);

%% concatenate

offset_features = cat(3, TL, TM, TR, ML, MM, MR, BL, BM, BR);


%% save

bfsave(offset_features, 'offset=200_feature-stack_f0000.tif')