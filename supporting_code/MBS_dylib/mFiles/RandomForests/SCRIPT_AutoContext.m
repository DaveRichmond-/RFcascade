%% SCRIPT calculate non-local features from feature stack

%% open feature stack

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack('subset_feature-stack_f0000.tif');

feature_stack = squeeze(imageStack);
clear imageStack size*

%% add position XY features

[X Y] = meshgrid([0:1023]);

feature_stack = cat(3,X,Y,feature_stack);

%% add contextual features

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack('test_probabilities.tif');

class_stack = squeeze(imageStack);
clear imageStack size*

%% calc features offset by 100 pixels, for 8 neighborhood

% user variables
offset = 100;

% create padded feature stack to sample from
pad_class_stack = padarray(class_stack,[offset offset 0],'replicate');

% initiate
TL = zeros(size(class_stack));
TM = zeros(size(class_stack));
TR = zeros(size(class_stack));
ML = zeros(size(class_stack));
MM = zeros(size(class_stack));
MR = zeros(size(class_stack));
BL = zeros(size(class_stack));
BM = zeros(size(class_stack));
BR = zeros(size(class_stack));

% fill top row
TL = pad_class_stack(1:end-2*offset, 1:end-2*offset, :);
TM = pad_class_stack(1:end-2*offset, offset+1:end-offset, :);
TR = pad_class_stack(1:end-2*offset, 2*offset+1:end, :);

% fill middle row
ML = pad_class_stack(offset+1:end-offset, 1:end-2*offset, :);
MM = pad_class_stack(offset+1:end-offset, offset+1:end-offset, :);
MR = pad_class_stack(offset+1:end-offset, 2*offset+1:end, :);

% fill middle row
BL = pad_class_stack(2*offset+1:end, 1:end-2*offset, :);
BM = pad_class_stack(2*offset+1:end, offset+1:end-offset, :);
BR = pad_class_stack(2*offset+1:end, 2*offset+1:end, :);

%% concatenate

offset_classes = cat(3, TL, TM, TR, ML, MM, MR, BL, BM, BR);

all_features = cat(3, feature_stack, offset_classes);

%% save

bfsave(all_features, 'AC_feature-stack_f0000.tif')