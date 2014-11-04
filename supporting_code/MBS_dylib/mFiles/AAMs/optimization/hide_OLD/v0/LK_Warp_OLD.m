function [st, s] = LK_Warp_OLD(s0,Sj,Sja,q,p);

% calculates a Warp based on the composition of a shape deformation (specified by p) followed by a similarity transformation (specified by q)
% st : a structure characterizing the thin-plate spline warp over the full domain, corresponding to the warp N(W(s0;p);q)
% s  : the final positions of the base-mesh vertices, after the composite warp is applied

% calculate position of base-mesh under Similarity Transformation

s_N = s0;

for i = 1:size(Sja,3),
    
    s_N = s_N + q(i)*Sja(:,:,i);
    
end

% TSP warp coefficients

[d ns0] = size(s0);
P = [s0.', ones(ns0,1)];

% QR decomposition of P     *** NOTE: DON'T NEED TO RECOMPUTE QR DECOMPOSITION EVERY ITERATION ***

[Q,R] = qr(P);
Q1 = Q(:, 1:d+1);
R1 = R(1:d+1,1:d+1);
Q(:, 1:d+1) = [];

K = stcol(s0, s0, 'tr');

%coefs1 = (s_N*Q/(Q'*K*Q))*Q'
%coefs2 = ((s_N - coefs1*K)*Q1)/(R(1:d+1,1:d+1).')

coefs2 = s_N*Q1/(R1.');

% apply TSP warp corresponding to coefs2, to the warped mesh: W(s0,p)

s_W = s0;

for i = 1:size(Sj,3),
    
    s_W = s_W + p(i)*Sj(:,:,i);
    
end

s = coefs2 * [s_W; ones(1,size(s0,2))];

% calc TSP warp corresponding to s0 -> s

st = tpaps(s0,s,1);