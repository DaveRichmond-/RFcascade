%% select variables by var_importance

feature_imp = sum(var_imp,1);

[feature_imp_sorted indx_sorted] = sort(feature_imp, 'descend');

figure,
plot((sum(feature_imp) - cumsum(feature_imp_sorted))/sum(feature_imp),'ro'),
axis([0 300 0 1])

% output labels

feat_keep = indx_sorted(1:50);
fileID = fopen('var_imp.txt','w')
fprintf(fileID,'%u\n', feat_keep);

% subset the feature stack
imageStack = imageStack(:,:,feat_keep([3:end]));

