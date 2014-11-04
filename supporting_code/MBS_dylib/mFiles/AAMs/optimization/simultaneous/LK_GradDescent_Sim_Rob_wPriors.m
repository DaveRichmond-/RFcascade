function [q_final,p_final, L_final, sp_final, Err_final, rho_p_final, q, p, L, sp_save, conv_flag] = LK_GradDescent_Sim_Rob_wPriors(raw_image,A0_vec,Grad_A0_vec,Ai_vec,Grad_Ai_vec,dNdq,dWdp,...
    s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,SFP_positions,pixelList,q_init,p_init,L_init,thresh,num_iters,reg_weights,step_size,sigma)

%

% initialize
conv_flag = 0;
w_image_vec = NaN(size(A0_vec));
c = sqrt(reg_weights);      % simpler to use the sqrt(reg_weight) for the math... 
q_prior = q_init;           % THIS IS REDUNDANT.  WILL ALWAYS INITIATE AT THE SAME VALUE AS THE PRIOR.  DROP ONE OF THESE.
p_prior = p_init;
L_prior = L_init;

q = NaN(size(q_init,1),num_iters);
dq = NaN(size(q_init,1),num_iters);
qp = NaN(size(q_init,1),num_iters);

p = NaN(size(p_init,1),num_iters);
dp = NaN(size(p_init,1),num_iters);
pp = NaN(size(p_init,1),num_iters);

L = NaN(size(L_init,1),num_iters);
dL = NaN(size(L_init,1),num_iters);
Lp = NaN(size(L_init,1),num_iters);

% iniate model
q(:,1) = q_init;
p(:,1) = p_init;
L(:,1) = L_init;

% grad descent loop
for i = 1:num_iters,

    % (1) Warp I with W(x;p) followed by N(x;q); where p,q come from last iteration ---------------->
    
    % calculate TSP warp from movement of base-mesh points under p, q
    [st1, s1] = LK_Warp(s0,Sj,Sja,weights,q(:,i),p(:,i));
    
    w_positions = fnval(st1, SFP_positions);
    
    if sum((w_positions(1,:) < 1) + (w_positions(2,:) < 1) + (w_positions(1,:) > size(raw_image,1)) + (w_positions(2,:) > size(raw_image,2))) > 0,
        q = q(:,1:i-1);
        p = p(:,1:i-1);
        L = L(:,1:i-1);
        
        q_final = q(:,i-1);
        p_final = p(:,i-1);
        L_final = L(:,i-1);
        Err_final = Err_image_vec;
        rho_p_final = rho_p;
        sp_final = sp;
        
        conv_flag = 1;
        display('template has left the image domain')
        return
    end
    
    for m = 1:length(pixelList),
        w_image_vec(m) = bilinear_interp(raw_image, w_positions(2,m), w_positions(1,m));
    end
    
    % normalize w_image_vec with respect to first guess of position
    if i==1,
        w_image_vec_MEAN = repmat(mean(w_image_vec), [size(w_image_vec,1),1]);
        w_image_vec_STD = repmat(sqrt(sum(w_image_vec.^2,1)), [size(w_image_vec,1),1]);
    end
    w_image_vec = w_image_vec - w_image_vec_MEAN;
    w_image_vec = w_image_vec ./ w_image_vec_STD;
    
    % (2) Compute the error image
    Err_image_vec =  A0_vec + Ai_vec*L(:,i) - w_image_vec;         % note the change of sign from before!
    
    % Compute the derivative of the robust error function, for (6)
    
    % square the error term.  Esq = t (from above)
    Esq = Err_image_vec.*Err_image_vec;
    [rho_p] = robust_error(Esq, sigma);
    
    % (5) compute the SD image
    SD = LK_SDimage_Sim_FAST(Grad_A0_vec, dNdq, dWdp, Ai_vec, L(:,i), Grad_Ai_vec);
    
    % pre-compute SD_rho_p, the combination of SD and rho_p that shows up multiple times
    SD_rho_p = repmat(rho_p,[1,size(SD,2)]).*SD;
    
    % (6) compute the Hessian
    H = LK_Hessian_Robust(SD, SD_rho_p);
    
    % <NEW>
    
    if i==1,
        
        inv_H = H^-1;
        SD_update = (SD_rho_p)' * Err_image_vec;
        
    else,
        
        d = [qp(:,i-1); pp(:,i-1); Lp(:,i-1)] - [q_prior; p_prior; L_prior];
        
        % (2 cont'd) Compute F(p)
        G = sqrt((c.*c)'*(d.*d));
        
        % (5a) Compute dF/dP and dp'/dDp
        if G == 0,
            dGdp = zeros(size(d));        % c.*sign(d);                                      % JUST RETURNS ZERO
        else
            dGdp = ((c.*c).*d) / G;
        end
        
        % reparameterization
        dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q(:,i-1),p(:,i-1),dq(:,i-1),dp(:,i-1),qp(:,i-1),pp(:,i-1));
        dppdDp = [dppdDp, zeros(size(dppdDp,1),length(L_prior)); zeros(length(L_prior),size(dppdDp,2)), eye(length(L_prior))];
        
        % (6a) calc Hessian and invert
        H_prior = (dGdp'*dppdDp)'*(dGdp'*dppdDp);
        inv_H = (H + H_prior)^-1;

        % (7) Compute SD update
        SD_update = (SD_rho_p'*Err_image_vec) + (dGdp'*dppdDp)'*G;        % MINUS SIGNS CANCEL ON PRIOR TERM
        
    end
    
    % </NEW>
    
    % (8) Compute the parameter updates: (dq,dp)
    Delta   = -inv_H * SD_update;                                     % MINUS SIGNS CANCEL ON PRIOR TERM
    dq(:,i) = Delta(1:4);
    dp(:,i) = Delta(5:4+size(p_init,1));
    dL(:,i) = Delta(4+size(p_init,1)+1:end);
    
    % shrink the step size to avoid oscillations                        
    dq(:,i) = dq(:,i).*step_size(1:4);
    dp(:,i) = dp(:,i).*step_size(5:4+size(p_init,1));
    dL(:,i) = dL(:,i).*step_size(4+size(p_init,1)+1:end);
    
    % (9) Update parameters by (dq,dp,dL)
    [qp(:,i), pp(:,i), sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q(:,i),p(:,i),dq(:,i),dp(:,i));
    Lp(:,i) = L(:,i) + dL(:,i);
    
    % re-assign for next iteration
    q(:,i+1) = qp(:,i);
    p(:,i+1) = pp(:,i);
    L(:,i+1) = Lp(:,i);
    
    % 
    if i == 1,
        sp_save(:,:,i) = s1;
        sp_save(:,:,i+1) = sp;
    else,
        sp_save(:,:,i+1) = sp;
    end
    
    % parameter based convergence test
    diff_q = q(:,i+1) - q(:,i);
    diff_p = p(:,i+1) - p(:,i);
    diff_L = L(:,i+1) - L(:,i);
    
    test_conv = abs([diff_q; diff_p; diff_L]) < thresh;
    
    if prod(test_conv),
        
        q = q(:,1:i+1);
        p = p(:,1:i+1);
        L = L(:,1:i+1);
        
        q_final = q(:,i+1);
        p_final = p(:,i+1);
        L_final = L(:,i+1);
        sp_final = sp;
        Err_final = Err_image_vec;
        rho_p_final = rho_p;

        i,
        return
        
    elseif i == num_iters-1,
        
        q_final = q(:,i+1);
        p_final = p(:,i+1);
        L_final = L(:,i+1);
        sp_final = sp;
        Err_final = Err_image_vec;
        rho_p_final = rho_p;
        
        conv_flag = 2;
        display(strcat('doesnt converge in ',num2str(num_iters-1),' iterations')),
        return
        
    end
    
    % cost based convergence test
    %{
    cost(i) = sum(Err_image_vec.^2);
    
    if (i ~= 1) && ((cost(i-1) - cost(i)) < thresh),
        
        i
        q_final = qp(:,i);
        p_final = pp(:,i);
        sp_final = sp;
        return
        
    end
    %}
    
end