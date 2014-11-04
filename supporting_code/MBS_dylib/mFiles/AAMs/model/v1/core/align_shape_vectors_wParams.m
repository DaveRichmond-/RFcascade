function [x_aligned, T] = align_shape_vectors_wParams(x,xp);

% aligns the vector x to vector xp using similarity transform, and assuming each vector has zero sum
% intended for use with AAM models, see Cootes and Taylor

numpoints = size(xp,1)/2;

if isequal(x,xp),
    
    x_aligned = x;
    
else,
    
    % create matrix representation of x
    
    xmat = [x(1:numpoints), x(numpoints+1:end)];
    xpmat = [xp(1:numpoints), xp(numpoints+1:end)];
    
    % calculate params for similarity transformation
    
    a = dot(x,xp)/dot(x,x);
    b = (dot(xmat(:,1),xpmat(:,2)) - dot(xpmat(:,1),xmat(:,2)))/dot(x,x);
    T = [a, -b; b, a];
    
    x_aligned_mat = (T*xmat')';
    x_aligned = [x_aligned_mat(:,1); x_aligned_mat(:,2)];
    
end

%{
% normalize xvec according to tangent method

x_aligned = x_aligned / dot(x_aligned, xp);
%}