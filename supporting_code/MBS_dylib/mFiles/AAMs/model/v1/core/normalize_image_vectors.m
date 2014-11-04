function [g, gbar, D] = normalize_image_vectors(g, mean_indx);

% EDITED !!!

% old approach
%{
% make all image vectors zero-mean, and one-variance

g = g - repmat(mean(g,1), [size(g,1),1]);
g = g ./ repmat(sqrt(sum(g.^2,1)), [size(g,1),1]);
%}

% choose reference vector as initial mean, translate to mean = 0, then scale to norm = 1

gbar_curr = g(:,mean_indx);

if mean(gbar_curr) ~= 0,                                % re-align gbar_curr to be zero mean (in case I remove this step from earlier)
    
    gbar_curr = gbar_curr - mean(gbar_curr);        

end

gbar_curr = gbar_curr / norm(gbar_curr);

% record reference frame as gbar_0

gbar_0 = gbar_curr;

% iteratively align all image vectors with the mean (and recompute mean each time)

thresh = 1e-6;

D(1) = 2*thresh;
k = 1;

while D(k) > thresh,
    
    k = k+1;
    
    for i = 1:size(g,2),
        
        g(:,i) = align_image_vectors(g(:,i), gbar_curr);
    
    end
    
    % estimate new mean from g
    
    gbar_new = mean(g,2);
    
    % align and normalize new estimate of mean
    
    gbar_new = align_image_vectors(gbar_new, gbar_0);           % not sure if this is a good idea !!!!!!!!!!!!!!
    gbar_new = gbar_new / norm(gbar_new);
    
    % calculate difference for breaking out of while loop
    
    D(k) = norm(gbar_new - gbar_curr);
    
    % re-assign new estimate to current estimate for next loop
        
    gbar_curr = gbar_new;
    
end

gbar = gbar_curr;

% old approach
%{
gbar = mean(g,2);
%}