%% calc class labels from probability stacks

%% open prob stack

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack('predict_probabilities_avg.tif');

all_feature_stack = squeeze(imageStack);
clear imageStack size*

% reshape into 4 separate experiments
all_feature_stack = reshape(all_feature_stack,[1024, 1024, 4, 22]);
%all_feature_stack = permute(all_feature_stack, [1 2 4 3]);

%%

label_images = NaN(size(all_feature_stack,1), size(all_feature_stack,2), size(all_feature_stack,3));
max_images = NaN(size(label_images));

for i = 1:size(label_images,1),
    i
    for j = 1:size(label_images,2),
    
        for k = 1:size(label_images,3),
            
            [max_val(i,j,k) label_images(i,j,k)] = max(squeeze(all_feature_stack(i,j,k,:)));
        
        end
    end
end

%% save

bfsave(uint8(label_images), 'predicted_class.tif');
bfsave(max_val, 'probability_of_predicted_class.tif');