function [dWdp, dNdp] = LK_warpJacobian(X,Y,pixelList,s0,Sj,Sja);

% calculates the Jacobian of the warp matrix, dW(i)/dp(j), with respect to the n parameters, p(j)
% as input, it takes:
% X : mesh of x positions in Shape Free Patch
% Y : mesh of y positions in Shape Free Patch
% pixelList : vector index describing which pixels (within a 2x2 unit box) are in the domain of the Shape Free Patch
% s0: mean shape, where s0 = reshape(xbar_vector, [8,2])';
% Sj: shape eigenvectors, where Sj = permute(reshape(Psi, [8,2,16]),[2 1 3]);

% calculate/construct some useful variables

[d ns0] = size(s0);
P = [s0.', ones(ns0,1)];

% initialize

dWdp = NaN(length(pixelList), d, size(Sj,3));
dNdp = NaN(length(pixelList), d, size(Sja,3));

% QR decomposition of P

[Q,R] = qr(P);
Q1 = Q(:, 1:d+1);
Q(:, 1:d+1) = [];

% calculate colocation matrix, K

K = stcol(s0, s0, 'tr');

% calculate A,C which act like L^-1, decomposed into affine and non-affine transformations

C = (Q / (Q'*K*Q)) * Q';                                    % coefficients of affine trans

A = ((eye(size(C*K)) - C*K)*Q1) / (R(1:d+1, 1:d+1).');      % coefficients of non-affine trans

% loop over domain of Shape Free Patch

for pid = 1:length(pixelList),
    
    % x,y value of ith pixel
    
    x = X(pixelList(pid));
    y = Y(pixelList(pid));
    
    U = stcol(s0, [x;y], 'tr');
    
    E = A*[x;y;1] + C*U;
    
    for i = 1:d,
        
        for j = 1:size(Sj,3),
            
            dWdp(pid,i,j) = dot(E,Sj(i,:,j));
            
        end
        
        for j = 1:size(Sja,3),
            
            dNdp(pid,i,j) = dot(E,Sja(i,:,j));
            
        end
        
    end
    
end