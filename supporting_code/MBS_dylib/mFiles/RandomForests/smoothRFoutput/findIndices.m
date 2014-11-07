function [indx] = findIndices(dataPath, fname_list)

%

load(strcat(dataPath, '/fname_complete_list.mat'));

% find index of each element in fname_list
indx = zeros(1,length(fname_list));
for i = 1:length(fname_list)    
    tmp = find(strcmp(fname_list{i},fname_complete_list));
    if (isempty(tmp)),
        display(strcat(fname_list{i},' not found in complete list'))
        indx(i) = [];
    else
        indx(i) = tmp;
    end
end

% debug warning.
if (length(unique(indx)) ~= length(indx))
    display(strcat('repeated filename in "buildRigidBackboneModel.m".  Number of unique filenames is: ',num2str(length(unique(indx)))))
    indx = unique(indx);
end