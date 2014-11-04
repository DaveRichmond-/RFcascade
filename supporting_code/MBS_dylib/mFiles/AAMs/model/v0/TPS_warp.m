function [x_prime] = TPS_warp(w, x_control, x_sample, sigma);

% 

% input
% w         : these are the weights that parameterize the warp
% x_control : these are the control points, that were already used to calculate the warp weights
% x_sample  : these are the sample points that you want to warp to x_prime
% sigma     : the stiffness of the plate

% output
% x_prime   : these are the warped points

% note, all arrays of points are arrays of 2d vectors of the form [x1, y1; x2, y2; ... ; xn, yn]


num_sample_points = size(x_sample,1);
num_control_points = size(x_control, 1);


for i = 1 : num_sample_points,
    
    % construct U(x)
    
    for j = 1 : num_control_points,
        
        U_vec(j) = TPS(x_sample(i,:), x_control(j,:), sigma);
        
    end
    
    U_vec(num_control_points + 1 : num_control_points + 3) = [1; x_sample(i,1); x_sample(i,2)];
    U_vec = U_vec(:);
    
    % calculate x_prime
    
    x_prime(i, :) = (w * U_vec)';
    
end