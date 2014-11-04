function [q,p] = test_position_priors(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,c,q_init,p_init,dq_init,dp_init,pos_nums,rc)

% update for loop
q(:,1) = q_init;
p(:,1) = p_init;

dq(:,1) = dq_init;
dp(:,1) = dp_init;

% calc update
[qp_init, pp_init, sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q(:,1),p(:,1),dq(:,1),dp(:,1));

qp(:,1) = qp_init(:);
pp(:,1) = pp_init(:);

q(:,2) = qp(:,1);
p(:,2) = pp(:,1);

%
for i=2:30,
    
    % current position
    [st, s] = LK_Warp(s0,Sj,Sja,weights,q(:,i),p(:,i));
    
    %
    for j=1:3,
        [F(j), dFdp(:,j)] = calc_dFdqp_POS(s(:,pos_nums(j)),rc(:,j),c,q(:,i),p(:,i),weights,s0,Sj,Sja,pos_nums(j));
    end
    F = F(:);
    
    % reparameterization
    dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q(:,i-1),p(:,i-1),dq(:,i-1),dp(:,i-1),qp(:,i-1),pp(:,i-1));
    
    % (6a) calc Hessian and invert
    H_prior = (dFdp'*dppdDp)'*(dFdp'*dppdDp);
    inv_H = (eye(size(H_prior)) + H_prior)^-1;
    
    % (7) Compute SD update
    SD_update = -(dFdp'*dppdDp)'*F;
    
    % (8) Compute the parameter updates: (dq,dp)
    Delta = inv_H * SD_update;
    dq(:,i) = Delta(1:4);
    dp(:,i) = Delta(5:end);
    
    dq = dq*0.3;
    dp = dp*0.3;
    
    % (9) Update parameters by (dq,dp)
    [qp(:,i), pp(:,i), sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q(:,i),p(:,i),dq(:,i),dp(:,i));
    
    % reset for next loop
    q(:,i+1) = qp(:,i);
    p(:,i+1) = pp(:,i);

end