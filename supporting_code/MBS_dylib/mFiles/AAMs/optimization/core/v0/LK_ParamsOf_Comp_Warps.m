function [qp, pp, sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q,p,dq,dp)

% calculate parameters, qp and pp, such that N(W(x;pp)qp) is equivalent to a composition of warps, N(W(x;p)q) . N(W(x;-dp);-dq)
% where -dp and -dq result from linear approx to N(W(x;dp);dq)^-1
% NOTE: PUTTING THIS IN DEDICATED FUNCTION MEANS RECALCULATING SOME OF THE WARPS IN LK_COMPWARPS

% reshape some things:

s0_vec  = reshape(permute(s0, [2 1]),[prod(size(s0)) 1]);
Sja_vec = squeeze(reshape(permute(Sja,[2 1 3]),[prod(size(s0)) 1 size(Sja,3)]));
Sj_vec  = squeeze(reshape(permute(Sj,[2 1 3]),[prod(size(s0)) 1 size(Sj,3)]));

% calculate destination of base-mesh vertices (s0 -> sp) under the composition of warps

sp = LK_CompWarps(s0,Sj,Sja,weights,q,p,dq,dp);
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