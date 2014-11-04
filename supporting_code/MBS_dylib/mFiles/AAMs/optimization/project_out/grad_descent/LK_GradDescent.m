function [] = LK_GradDescent()

% <NEW>
% user set parameters
q_weights = [1e6; 0; 0; 0];
p_weights = zeros(length(p_init),1);
reg_weight = [q_weights; p_weights];  % cost, i.e. scaling of regularization term
% </NEW>

clear cost sp_save store_dp store_dq

% initialize
thresh = eps;
w_image_vec = NaN(size(A0_vec));
q_old = q_init;
p_old = p_init;
q0 = q_init;
p0 = p_init;
c = sqrt(reg_weight);   % simpler to use the sqrt(reg_weight) for the math... 

for t = 1:1%size(Trev_imageStack,3),
    
    t,
    
    % step frame #
    raw_image = Trev_imageStack(:,:,t);

    % grad descent loop
    for i = 1:2,%100,
        
        % (1) Warp I with W(x;p) followed by N(x;q); where p,q come from last iteration ---------------->
        
        % calculate TSP warp from movement of base-mesh points under p_old, q_old
        [st1, s1] = LK_Warp(s0,Sj,Sja,weights,q_old,p_old);
        
        % store shape during grad descent
        %{
        if i == 1,
            sp_save(:,:,i) = s1;
        end
        %}
        
        w_positions = fnval(st1, SFP_positions);
        
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
        Err_image_vec = w_image_vec - A0_vec;
        
        % <NEW>
        
        d = [q_old; p_old] - [q0; p0];
        
        % (2 cont'd) Compute F(p)
        F = sqrt((c.*c)'*(d.*d));
        
        % (5a) Compute dF/dP and dp'/dDp
        if F == 0,
            dFdp = zeros(size(d));        % c.*sign(d);                                      % JUST RETURNS ZERO
        else
            dFdp = ((c.*c).*d) / F;
        end
        
        % calc reparameterization
        %dppdDp = diag(-[1 1 1 1 1]);                                 % LINEAR APPROX FOR NOW
        if i==1,
            store_p_old = p_old';
            store_q_old = q_old';
            store_dq = zeros(size(q_init))';
            store_dp = zeros(size(p_init))';
        end
        dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,store_q_old,store_p_old,store_dq,store_dp,q_old,p_old);
        
        % (6a) calc Hessian and invert
        H_prior = (dFdp'*dppdDp)'*(dFdp'*dppdDp);
        inv_H = (H + H_prior)^-1;
        
        % (7) Compute SD update
        SD_update = SD' * Err_image_vec;
        SD_update = SD_update - (dFdp'*dppdDp)'*F;
        
        % </NEW>        
        
        % (8) Compute the parameter updates: (dq,dp)
        Delta_qp = inv_H * SD_update;                                   % MINUS SIGN MISSING ???
        dq = Delta_qp(1:4);
        dp = Delta_qp(5:end);
        
        % store updates for troubleshooting
        %{
    store_dq(:,i) = dq(q_indx)';
    store_dp(i) = dp(p_indx)';
        %}
        
        % (9) Update parameters by (dq,dp)
        [q_new, p_new, sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q_old,p_old,dq,dp);
        
        % store for troubleshooting
        store_q_old = q_old;
        store_p_old = p_old;
        store_dq = dq;
        store_dp = dp;
        %{
    diff_q = q_new' - q_old;
    diff_p = p_new' - p_old;
    
    store_diff_qp(:,i) = [diff_q; diff_p];
    
    
    % test convergence
    diff_q = abs(q_new' - q_old);
    diff_p = abs(p_new' - p_old);
        %}
        
        % re-assign for next iteration
        q_old = q_new';
        p_old = p_new';
        
        % store position
        %sp_save(:,:,i+1) = sp;
        cost(i) = sum(Err_image_vec.^2);
        
        if (i ~= 1) && ((cost(i-1) - cost(i)) < thresh),
            break
        end
        
    end
    
    % save q,p
    q_save(:,t) = q_old;
    p_save(:,t) = p_old;
    sp_save(:,:,t) = sp;
    cost_save(:,t) = cost(i);
    
    % update prior for next time point
    q0 = q_save(:,t);
    p0 = p_save(:,t);
    
end
