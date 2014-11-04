clear all
load('precomputed.mat')

%%

c = [1e3;1e3;1e3;1e3;1e3];

q_prior = [1;1;1;1];
p_prior = 0.1;
q_init = q_prior;
p_init = p_prior;
dq_init = [0.1;0.1;0.1;0.1];
dp_init = 0.05;

[q,p,dq,dp,qp,pp] = test_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,c,q_prior,p_prior,q_init,p_init,dq_init,dp_init)