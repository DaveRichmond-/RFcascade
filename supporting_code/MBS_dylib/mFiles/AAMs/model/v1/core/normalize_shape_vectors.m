function [xvec, xbar, D] = normalize_shape_vectors(xvec, mean_indx);

%

% choose reference vector as initial mean and scale to norm = 1

xbar_curr = xvec(:,mean_indx);
xbar_curr = xbar_curr / norm(xbar_curr);

% record reference frame as xbar_0

xbar_0 = xbar_curr;


% old approach
%{
% normalize all xvec's against reference

xvec = xvec / norm(xvec(:,mean_indx));
xbar_curr = xvec(:,mean_indx);
%}

% iterative alignment of all vectors with mean, to define mean

thresh = 1e-6;

D = 2*thresh;
k = 1;

while D(k) > thresh,
    
    k = k+1;
    
    % align all vectors to current estimate of mean
    
    for j=1:size(xvec,2),
        
        xvec(:,j) = align_shape_vectors(xvec(:,j), xbar_curr);
        
        % normalize xvec according to tangent method
    
        xvec(:,j) = xvec(:,j) / dot(xvec(:,j), xbar_curr);
        
    end
    
    % estimate new mean from xvec
    
    xbar_new = mean(xvec,2);
    
    % align and normalize new estimate of mean
    
    xbar_new = align_shape_vectors(xbar_new, xbar_0);
    xbar_new = xbar_new / norm(xbar_new);
    
    % calculate difference, for breaking out of while loop
    
    D(k) = norm(xbar_new - xbar_curr);
    
    % re-assign new estimate to current estimate for next loop
    
    xbar_curr = xbar_new;
    
end

xbar = xbar_curr;

% old approach
%{
xbar = mean(xvec, 2);
%}