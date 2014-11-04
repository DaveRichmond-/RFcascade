%% script to test making jacobian of warp for thin plate spline and AAM

% inputs

x1 = 0;     % x-coord of position to be warped
x2 = 0;     % y-coord of position to be warped

s0 = shape(1).xbar_vector;      % average shape
s  = shape(1).Psi;              % shape vectors, sj

x = reshape(s0, [8,2])';

% characteristic matrices of the transformation

[m ns0] = size(s0);

P = [s0.', ones(ns0, 1)];

A = 

for i = 1:2,
    
    for j = 1:8,
        
        for k = 1:8,
            
            
        
        term1 = 