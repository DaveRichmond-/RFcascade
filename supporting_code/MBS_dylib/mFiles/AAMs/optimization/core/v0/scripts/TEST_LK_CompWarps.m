%% test LK_CompWarps

% script to test composition of warps by linear approximation to inverse

%%

scale = 1e-3;

q = scale*[1 1 1 1];
p = scale*[1 1 1 1];

dq = q;
dp = p;

[qp, pp, sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q,p,dq,dp);

qp ./ q,
pp ./ p,