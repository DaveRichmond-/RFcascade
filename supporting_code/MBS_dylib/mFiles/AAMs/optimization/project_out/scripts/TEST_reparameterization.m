function [q,p,dq,dp,qp,pp] = test_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,c,q_prior,p_prior,q_init,p_init,dq_init,dp_init)

% calc update
[qp_init, pp_init, sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q_init,p_init,dq_init,dp_init);

qp_init = qp_init(:);
pp_init = pp_init(:);

q(:,2) = qp_init;
p(:,2) = pp_init;

% update for loop
q(:,1) = q_init;
p(:,1) = p_init;

dq(:,1) = dq_init;
dp(:,1) = dp_init;

qp(:,1) = qp_init;
pp(:,1) = pp_init;

for i=2:20,
    
    % calc new updates
    
    d = [qp(:,i-1); pp(:,i-1)] - [q_prior; p_prior];
    
    % (2 cont'd) Compute F(p)
    F = sqrt((c.*c)'*(d.*d));
    
    % (5a) Compute dF/dP and dp'/dDp
    if F == 0,
        dFdp = zeros(size(d));        % c.*sign(d);                                      % JUST RETURNS ZERO
    else
        dFdp = ((c.*c).*d) / F;
    end
    
    % reparameterization
    %dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q(:,i-1),p(:,i-1),dq(:,i-1),dp(:,i-1),qp(:,i-1),pp(:,i-1));
    dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q(:,i-1),p(:,i-1),dq(:,i-1),dp(:,i-1),qp(:,i-1),pp(:,i-1));
    
    % (6a) calc Hessian and invert
    H_prior = (dFdp'*dppdDp)'*(dFdp'*dppdDp);
    inv_H = (eye(size(H_prior)) + H_prior)^-1;
    
    % (7) Compute SD update
    SD_update = -(dFdp'*dppdDp)'*F;
    
    % (8) Compute the parameter updates: (dq,dp)
    Delta = inv_H * SD_update;                                   % MINUS SIGN MISSING ???
    dq(:,i) = Delta(1:4);
    dp(:,i) = Delta(5:end);
    
    % (9) Update parameters by (dq,dp)
    [qp(:,i), pp(:,i), sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q(:,i),p(:,i),dq(:,i),dp(:,i));
    
    % reset for next loop
    q(:,i+1) = qp(:,i);
    p(:,i+1) = pp(:,i);

end
