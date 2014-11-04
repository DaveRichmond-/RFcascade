function [g] = align_image_vectors(g_im, gbar);

% aligns g_im onto gbar, according to Cootes and Taylor

g = (g_im - mean(g_im)) / dot(g_im, gbar);                                                % uses tangent projection approach