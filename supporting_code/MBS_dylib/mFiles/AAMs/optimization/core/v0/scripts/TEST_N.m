%% test N(x;q)

%%

%

s0 = reshape(shape(1).xbar_vector, [8,2])';
Sj = permute(reshape(shape(1).Psi, [8,2,16]), [2 1 3]);

% make corresponding e-vectors of Similarity Transform

R = [0 -1; 1 0];

Sja(:,:,1) = s0;
Sja(:,:,2) = R * s0;
Sja(:,:,3) = [ones(1,size(s0,2)); zeros(1,size(s0,2))];
Sja(:,:,4) = [zeros(1,size(s0,2)); ones(1,size(s0,2))];

% vector representations                    IN FUTURE, WORK ONLY WITH VECTOR REPRESENTATION.  YOU RARELY NEED MATRIX REPRESENTATION!

s0_vec  = reshape(permute(s0, [2 1]),[16 1]);
Sj_vec  = squeeze(reshape(permute(Sj,[2 1 3]),[16 1 16]));
Sja_vec = squeeze(reshape(permute(Sja,[2 1 3]),[16 1 4]));

% orthonormalize Sja_vec

[Q,R] = qr(Sja_vec,0);
weights = diag(R);

Sja_vec = Q;

% map back to Sja

Sja = permute(reshape(Sja_vec,[8,2,4]),[2 1 3]);

%%

q = [0 0 2 3];         % coefficients of similarity transformation
p = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];


%% test

tic,
[st s_method1] = LK_Warp2(s0,Sj,Sja,q,weights,p)
toc,
tic,
[st s_method2] = LK_Warp(s0,Sj,Sja,q,p)
toc,


