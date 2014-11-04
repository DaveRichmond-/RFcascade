function [A,t] = LK_qtoA(q, weights)

% convert from parameters of similarity transform, q, to parameters of corresponding affine transform, A and t

a  = q(1)/weights(1);
b  = q(2)/weights(2);
tx = q(3)/weights(3);
ty = q(4)/weights(4);

A = [(1+a), -b; b, (1+a)];
t = [tx; ty];
