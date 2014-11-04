function [q_final,p_final, sp_final, q, p, sp_save] = LK_HierGradDescent_wPriors(raw_image,A0_vec,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,SFP_positions,pixelList,H,SD,q_init,p_init,thresh,num_iters,reg_weights,step_size,hierStruct)

%

% initialize
w_image_vec = NaN(size(A0_vec));
input_c = sqrt(reg_weights); 
q_prior = q_init;           % THIS IS REDUNDANT.  WILL ALWAYS INITIATE AT THE SAME VALUE AS THE PRIOR.  DROP ONE OF THESE.  IF CHANGED, CAREFUL WITH FORCING PRIOR INFLUENCE TO ZERO IN FIRST LOOP BELOW
p_prior = p_init;

q = NaN(size(q_init,1),num_iters);
dq = NaN(size(q_init,1),num_iters);
qp = NaN(size(q_init,1),num_iters);

p = NaN(size(p_init,1),num_iters);
dp = NaN(size(p_init,1),num_iters);
pp = NaN(size(p_init,1),num_iters);

% iniate model
q(:,1) = q_init;
p(:,1) = p_init;

k = 1;

for j = 1:size(hierStruct,2),
    
    % turn off regularization for all but the variable(s) that are dictated by hierStruct
    c = input_c.*hierStruct(:,j);
    
    % grad descent loop
    for i = 1:num_iters,
        
        %k
        % (1) Warp I with W(x;p) followed by N(x;q); where p,q come from last iteration ---------------->
        
        % calculate TSP warp from movement of base-mesh points under p, q
        [st1, s1] = LK_Warp(s0,Sj,Sja,weights,q(:,k),p(:,k));
        
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
        
        if k==1,
            
            inv_H = H^-1;
            
            % update
            SD_update = SD' * Err_image_vec;
            
        else,
            
            d = [qp(:,k-1); pp(:,k-1)] - [q_prior; p_prior];
            
            % (2 cont'd) Compute F(p)
            F = sqrt((c.*c)'*(d.*d));
            
            % (5a) Compute dF/dP and dp'/dDp
            if F == 0,
                dFdp = zeros(size(d));
            else
                dFdp = ((c.*c).*d) / F;
            end
            
            % reparameterization
            dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q(:,k-1),p(:,k-1),dq(:,k-1),dp(:,k-1),qp(:,k-1),pp(:,k-1));
            
            % (6a) calc Hessian and invert
            H_prior = (dFdp'*dppdDp)'*(dFdp'*dppdDp);

            % invert H
            %inv_H = (eye(size(H)).*repmat(diag(H + H_prior),[1 size(H,2)]))^-1;
            inv_H = (H + H_prior)^-1;
            
            % (7) Compute SD update
            SD_update = SD' * Err_image_vec;
            SD_update = SD_update - (dFdp'*dppdDp)'*F;
            
        end
        
        % </NEW>
        
        % (8) Compute the parameter updates: (dq,dp)
        Delta = inv_H * SD_update;
        Delta = Delta(:);
        
        % select some dimensions according to hierStruct
        dq(:,k) = hierStruct(1:4,j).*Delta(1:4);
        dp(:,k) = hierStruct(5:end,j).*Delta(5:end);
        
        % shrink the step size to avoid oscillations
        dq(:,k) = dq(:,k).*step_size(1:4);
        dp(:,k) = dp(:,k).*step_size(5:end);
        
        % (9) Update parameters by (dq,dp)
        [qp(:,k), pp(:,k), sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q(:,k),p(:,k),dq(:,k),dp(:,k));
        
        % re-assign for next iteration
        
        for hS_indx = 1:4,
            if hierStruct(hS_indx,j) == 1,
                q(hS_indx,k+1) = qp(hS_indx,k);
            else
                q(hS_indx,k+1) = q(hS_indx,k);
            end
        end
        
        for hS_indx = 5:size(hierStruct,1),
            if hierStruct(hS_indx,j) == 1,
                p(hS_indx-4,k+1) = pp(hS_indx-4,k);
            else
                p(hS_indx-4,k+1) = p(hS_indx-4,k);
            end
        end
        
        %{
        q(:,k+1) = qp(:,k);
        p(:,k+1) = pp(:,k);
        %}
        
        %
        if k == 1,
            sp_save(:,:,k) = s1;
            sp_save(:,:,k+1) = sp;
        else,
            sp_save(:,:,k+1) = sp;
        end
        
        % parameter based convergence test
        diff_q = qp(:,k) - q(:,k);
        diff_p = pp(:,k) - p(:,k);
        
        test_conv = abs([diff_q; diff_p]) < thresh;
        
        if prod(test_conv),
            k,
            q_final = qp(:,k);
            p_final = pp(:,k);
            sp_final = sp;
            return
            
        elseif k==num_iters,
            error('doesnt converge')
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
        
        % save lots of stuff for troubleshooting
        %{
    store_q = q;
    store_p = p;
    store_dq = dq;
    store_dp = dp;

    diff_q = q_new' - q;
    diff_p = p_new' - p;
    
    store_diff_qp(:,i) = [diff_q; diff_p];
    
    
    % test convergence
    diff_q = abs(q_new' - q);
    diff_p = abs(p_new' - p);

    store_dq(:,i) = dq(q_indx)';
    store_dp(i) = dp(p_indx)';
    
        %}
        
        k = k+1;
        
    end
    
end

q_final = qp(:,size(hierStruct,2)*num_iters);
p_final = pp(:,size(hierStruct,2)*num_iters);
sp_final = sp;