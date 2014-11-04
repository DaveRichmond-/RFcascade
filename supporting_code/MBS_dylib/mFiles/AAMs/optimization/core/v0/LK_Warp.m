function [st, s] = LK_Warp(s0,Sj,Sja,weights,q,p);

% calculates a Warp based on the composition of a shape deformation (specified by p) followed by a similarity transformation (specified by q)
% st : a structure characterizing the thin-plate spline warp over the full domain, corresponding to the warp N(W(s0;p);q)
% s  : the final positions of the base-mesh vertices, after the composite warp is applied

% calculate coefs of similarity transform, based on q and weights

a = q(1)/weights(1);
b = q(2)/weights(2);
tx = q(3)/weights(3);
ty = q(4)/weights(4);

A = [(1+a), -b, tx; b, (1+a), ty];                          % consider just passing in A and t, not Sja, weights, q

% apply warp, W, to base mesh, s0

s_W = s0;

for i = 1:size(Sj,3),
    
    s_W = s_W + p(i)*Sj(:,:,i);
    
end

% apply similarity transform, N, to warped base-mesh, s

s = A * [s_W; ones(1,size(s0,2))];

% calc TSP warp corresponding to s0 -> s

st = tpaps(s0,s,1);