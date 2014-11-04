%% SCRIPT calculate non-local features from feature stack

%% open feature stack

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack('predict_probabilities.tif');

all_feature_stack = squeeze(imageStack);
clear imageStack size*

% reshape into 4 separate experiments
all_feature_stack = reshape(all_feature_stack,[1024, 1024, 4, 22]);
all_feature_stack = permute(all_feature_stack, [1 2 4 3]);

%% merge from 5 separate results

num_forests = 5;

for i = 1:num_forests,

    [imageStack,sizeC,sizeZ,sizeT] = bf_openStack(strcat('rf',num2str(i),...
        '/rf',num2str(i),'.tif'));
    
    all_feature_stack = squeeze(imageStack);
    clear imageStack size*

    % reshape into 4 separate experiments
    all_feature_stack = reshape(all_feature_stack,[1024, 1024, 4, 22]);
    all_feature_stack = permute(all_feature_stack, [1 2 4 3]);
    
    % rename
    store_all(:,:,:,:,i) = all_feature_stack;
    clear all_feature_stack

end

all_feature_stack = squeeze(mean(store_all,5));
clear store_all i num_forests

%% save all_feature_stack as .tif to replace 5 separate probability stacks

predict_probabilities_save = permute(all_feature_stack, [1 2 4 3]);
predict_probabilities_save = reshape(predict_probabilities_save, [1024 1024 88]);

bfsave(predict_probabilities_save, 'predict_probabilities_avg.tif')

%% calc features offset by x pixels, for 8 neighborhood

% user variables
offset = 50;

for stack_num = 1:4,
    
    clear feature_stack
    feature_stack = squeeze(all_feature_stack(:,:,:,stack_num));
    
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
    
    % concatenate
    if (offset == 25),
        offset_features = cat(3, TL, TM, TR, ML, MM, MR, BL, BM, BR);
    elseif (offset == 50),
        % for second set of offset (don't need to repeat the central point)
        offset_features = cat(3, TL, TM, TR, ML, MR, BL, BM, BR);
    end
    
    % save
    bfsave(offset_features, strcat('offset=',num2str(offset),'_prob_feature_stack',num2str(stack_num),'.tif'))
    
end