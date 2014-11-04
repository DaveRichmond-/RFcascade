function [b err] = fit_shape_model(x_vector, xbar_vector, Psi);

% fits a new shape, x, to a shape model characterized by Psi
% returns the parameterization of the shape, b, according to the best fit to the model

% first zero the shape vector x

xmat = reshape(x_vector, [size(x_vector,1)/2, 2]);
xmat = xmat - repmat(mean(xmat,1),[size(xmat,1) 1]);
x_vector = xmat(:);

% 

b = zeros(size(x_vector));
 
for k = 1:1e2,          %1e2 loops b/c typically converges by then
    
    x_inst = xbar_vector + Psi*b;
    
    x_aligned = align_shape_vectors(x_vector, x_inst);
    x_aligned = x_aligned / dot(x_aligned, xbar_vector);
    
    b = Psi' * (x_aligned - xbar_vector);

    % evaluate fit
    
    err(k) = norm(x_aligned - (xbar_vector + Psi*b));

end