function [U] = TPS(x, xi, sigma);

% calculates the function U(r) according to the Thin-Plate Spline model of deformation

% scalar distance between two points x and xi, where x and xi are in higher dimension (typically d=2)

r = norm(x - xi);

% TPS formula

if r == 0,
    
    U = 0;
    
else

    U = ((r / sigma)^2) * log(r / sigma);
    
end