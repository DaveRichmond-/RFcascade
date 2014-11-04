function [g, gbar, D] = normalize_image_vectors(g, mean_indx);

% make all image vectors zero-mean, and one-variance

g = g - repmat(mean(g,1), [size(g,1),1]);
g = g ./ repmat(sqrt(sum(g.^2,1)), [size(g,1),1]);

% set middle image vector as mean

gbar_curr = g(:,mean_indx);

% iteratively align all image vectors with the mean (and recompute mean each time)

thresh = 1e-6;

D(1) = 2*thresh;
k = 1;

while k<100,%D(k) > thresh,
    
    k = k+1;
    
    for i = 1:size(g,2),
        
        g(:,i) = align_image_vectors(g(:,i), gbar_curr);
    
    end
    
    gbar_new = mean(g,2);
    
    D(k) = norm(gbar_new - gbar_curr);
    gbar_curr = gbar_new;
    
end

gbar = mean(g,2);