function [Sja, Sja_vec, weights] = LK_SimTransfBasis(s0);

% calculate orthonormal basis vectors of similarity transform
% assumes size(s0) = [num_dims, num_points]

R = [0 -1; 1 0];

Sja(:,:,1) = s0;
Sja(:,:,2) = R * s0;
Sja(:,:,3) = [ones(1,size(s0,2)); zeros(1,size(s0,2))];
Sja(:,:,4) = [zeros(1,size(s0,2)); ones(1,size(s0,2))];

Sja_vec = squeeze(reshape(permute(Sja, [2 1 3]), [prod(size(s0)) 1 4]));

% orthonormalize Sja and Sja_vec

[Q,R] = qr(Sja_vec, 0);
weights = diag(R);

if weights(1) > 0,
    
    Sja_vec = Q;
    
else
    
    weights = -weights;
    Sja_vec = -Q;
    
end

Sja = permute(reshape(Sja_vec, [size(s0,2) size(s0,1) 4]), [2 1 3]);