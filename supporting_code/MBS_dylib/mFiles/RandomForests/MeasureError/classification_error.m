function [] = classification_error(true_labels, classification_labels);

% measure the error between a classification image, and the true labels

for i = 1:size(true_labels,1),
    
    for j = 1:size(true_labels,2),
        
        

