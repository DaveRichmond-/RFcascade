function [w] = calc_weights_TPS(x, xprime, sigma);

% calculates the weights corresponding to mapping the control points x onto corresponding points xprime.
% uses a thin-plate splin (TPS)
% sigma parameterizes the 'stiffness' of the thin plate
% assumes two dimensions
% format for x is [x1, y1; x2, y2; ....; xn, yn], same for xprime

K = zeros(size(x,1), size(xprime,1));

for i = 1:length(x),
    
    for j = 1:length(xprime),
        
        K(i,j) = TPS(x(i,:), xprime(j,:), sigma);
        
    end
    
end

Q = [ones(size(x,1),1), x(:,1), x(:,2)];

L = [K, Q; Q', zeros(3)];

w = inv(L') * [xprime; zeros(3,2)];
w = w';