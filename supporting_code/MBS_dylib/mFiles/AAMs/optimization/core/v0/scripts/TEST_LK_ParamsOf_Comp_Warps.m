%% script to build/test deriving equivalent parameter set from a composition of warps

%% specify parameters of individual warps (these will come from previous step in LK cycle).

q = [10 10 2 3];         % coefficients of similarity transformation
%corr = [-1 -1 -2.8284 -2.8284];
%q = q.*corr;
%p = shape(1).b(:,10)';  % coefficients of shape for 10th frame of somite 7
p = [1 3 4 8 0 0 0 0 0 0 0 0 0 0 0 0];

dq = [1 1 1 1];
%dq = dq.*corr;
dp = [0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.5];

%% 

% define characteristic matrices from model

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

% full code:

%{
% calculate destination of base-mesh vertices (s0 -> sp) under the composition of warps

[sp] = LK_CompWarps(s0,Sj,Sja,weights,q,p,dq,dp);
sp_vec = reshape(permute(sp,[2 1]),[16,1]);

% back-calculate qp's

for i = 1:size(Sja,3),
    
    qp(i) = dot(Sja_vec(:,i), (sp_vec - s0_vec));

end

% calc s = N(sp,q)^-1, the destination of sp under the inverse similarity transform

[Ap,tp] = LK_qtoA(qp, weights);

s = (Ap^-1)*(sp - repmat(tp,[1,size(s0,2)]));
s_vec = reshape(permute(s,[2 1]),[16 1]);

% back-calculate pp's

for i = 1:size(Sj,3),
    
    pp(i) = dot(Sj_vec(:,i), (s_vec - s0_vec));
    
end
%}

% function call

[qp, pp, sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q,p,dq,dp);

%% test: apply warp defined by (qp,pp) to s0.  should return sp (in a single step)

[tmp, test_sp] = LK_Warp(s0,Sj,Sja,weights,qp,pp);

sp,
test_sp,
dif = sum(sum(sp - test_sp)),