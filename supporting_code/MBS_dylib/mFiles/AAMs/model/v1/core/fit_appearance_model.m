function [b, err] = fit_appearance_model(g, gbar, Psi);

% fits a new appearance vector, g, to an appearance model characterized by Psi
% returns the parameterization of the shape, b, according to the best fit to the model

% make image vector zero-mean, and one-variance

g = g - mean(g,1);
g = g ./ sqrt(sum(g.^2,1));

% 

b = zeros(size(Psi,2),1);

for k = 1:1e2,          %1e2 loops b/c typically converges by then
    
    g = gbar + Psi*b;
    
    g_aligned = align_image_vectors(g, gbar);
    
    b = Psi' * (g_aligned - gbar);

    % evaluate fit
    
    err(k) = norm(g_aligned - (gbar + Psi*b));

end