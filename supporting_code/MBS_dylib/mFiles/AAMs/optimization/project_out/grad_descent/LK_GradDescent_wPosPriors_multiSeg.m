function [q_final, p_final, sp_final, q, p, sp_save, conv_flag] = LK_GradDescent_wPosPriors_multiSeg(raw_image,A0_vec,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,SFP_positions,pixelList,H,SD,...
    q_init,p_init,thresh,num_iters,reg_weights,step_size,reg_weights_pos)

% notes: change q_init, p_init to have N columns; rc shouldn't be necessary as an input; pos_nums not necessary

%

% initialize
conv_flag = 0;
w_image_vec = NaN(size(A0_vec));
c = sqrt(reg_weights);      % simpler to use the sqrt(reg_weight) for the math... 
c_pos = sqrt(reg_weights_pos);

q_prior = q_init;           % THIS IS REDUNDANT.  WILL ALWAYS INITIATE AT THE SAME VALUE AS THE PRIOR.  DROP ONE OF THESE.
p_prior = p_init;

q  = NaN(size(q_init,1),size(q_init,2),num_iters);
dq = NaN(size(q_init,1),size(q_init,2),num_iters);
qp = NaN(size(q_init,1),size(q_init,2),num_iters);

p  = NaN(size(p_init,1),size(p_init,2),num_iters);
dp = NaN(size(p_init,1),size(p_init,2),num_iters);
pp = NaN(size(p_init,1),size(p_init,2),num_iters);

% iniate model
q(:,:,1) = q_init;
p(:,:,1) = p_init;

% grad descent loop
for i = 1:num_iters,

    for seg_num = 1:size(q_init,2),
        
        % assign position numbers of points that are constrained in current segment
        if seg_num == 1,
            pos_nums = [3,4,5];
        elseif seg_num == size(q_init,2),
            pos_nums = [7,8,1];
        else
            pos_nums = [7,8,1,3,4,5];
        end
        
        % (1) Warp I with W(x;p) followed by N(x;q); where p,q come from last iteration ---------------->
        
        % calculate TSP warp from movement of base-mesh points under p, q
        [st, s(:,:,seg_num,i)] = LK_Warp(s0,Sj,Sja,weights,q(:,seg_num,i),p(:,seg_num,i));
        
        w_positions = fnval(st, SFP_positions);
        
        if sum((w_positions(1,:) < 1) + (w_positions(2,:) < 1) + (w_positions(1,:) > size(raw_image,1)) + (w_positions(2,:) > size(raw_image,2))) > 0,
            
            q = q(:,:,1:i-1);
            p = p(:,:,1:i-1);
            
            q_final = q(:,:,i-1);
            p_final = p(:,:,i-1);
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
        Err_image_vec = w_image_vec - A0_vec;
        
        % <NEW>
        
        if i==1,
            
            inv_H = H^-1;
            SD_update = SD' * Err_image_vec;
            
        else,
            
            % smooth parameter trajectory priors
            d = [qp(:,seg_num,i-1); pp(:,seg_num,i-1)] - [q_prior(:,seg_num); p_prior(:,seg_num)];
            
            % (2 cont'd) Compute F(p)
            F(1) = sqrt((c.*c)'*(d.*d));
            
            % (5a) Compute dF/dP and dp'/dDp
            if F(1) == 0,
                dFdp(:,1) = zeros(size(d));        % c.*sign(d);                                      % JUST RETURNS ZERO
            else
                dFdp(:,1) = ((c.*c).*d) / F(1);
            end
            
            % position priors
            
            % read in positions from neighboring somites
            if seg_num == 1,
                rc = s(:,[1,8,7],seg_num+1,i-1);
            elseif seg_num == size(q_init,2),
                rc = s(:,[5,4,3],seg_num-1,i-1);
            else
                rc = [s(:,[5,4,3],seg_num-1,i-1), s(:,[1,8,7],seg_num+1,i-1)];
            end
            
            %
            for j=1:size(rc,2),
                [F(j+1), dFdp(:,j+1)] = calc_dFdqp_POS(s(:,pos_nums(j),seg_num),rc(:,j),c_pos,q(:,seg_num,i),p(:,seg_num,i),weights,s0,Sj,Sja,pos_nums(j));
            end
            clear rc
            F = F(:);
            
            % reparameterization
            dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q(:,seg_num,i-1),p(:,seg_num,i-1),dq(:,seg_num,i-1),dp(:,seg_num,i-1),qp(:,seg_num,i-1),pp(:,seg_num,i-1));
            
            % (6a) calc Hessian and invert
            H_prior = (dFdp'*dppdDp)'*(dFdp'*dppdDp);
            inv_H = (H + H_prior)^-1;
            %inv_H = (eye(size(H_prior)) + H_prior)^-1;      % KILL IMAGE TERM
            
            % (7) Compute SD update
            SD_update = SD' * Err_image_vec;
            SD_update = SD_update - (dFdp'*dppdDp)'*F;
            %SD_update = - (dFdp'*dppdDp)'*F;               % KILL IMAGE TERM
            
        end
        
        % </NEW>
        
        % (8) Compute the parameter updates: (dq,dp)
        Delta = inv_H * SD_update;                                   % MINUS SIGN MISSING ???
        dq(:,seg_num,i) = Delta(1:4);
        dp(:,seg_num,i) = Delta(5:end);
        
        % shrink the step size to avoid oscillations                        % CAREFUL !!!
        dq(:,seg_num,i) = dq(:,seg_num,i).*step_size(1:4);
        dp(:,seg_num,i) = dp(:,seg_num,i).*step_size(5:end);
        
        % (9) Update parameters by (dq,dp)
        [qp(:,seg_num,i), pp(:,seg_num,i), sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q(:,seg_num,i),p(:,seg_num,i),dq(:,seg_num,i),dp(:,seg_num,i));
        
        % re-assign for next iteration
        q(:,seg_num,i+1) = qp(:,seg_num,i);
        p(:,seg_num,i+1) = pp(:,seg_num,i);
        
        %
        if i == 1,
            sp_save(:,:,seg_num,i) = s(:,:,seg_num,i);
            sp_save(:,:,seg_num,i+1) = sp;
        else,
            sp_save(:,:,seg_num,i+1) = sp;
        end
        
        % parameter based convergence test
        test_conv = 0;
        if seg_num == size(q_init,2),
            diff_q = q(:,:,i+1) - q(:,:,i);
            diff_p = p(:,:,i+1) - p(:,:,i);
        
            test_conv = abs([diff_q; diff_p]) < repmat(thresh,[1,size(q_init,2)]);
        end
        
        if prod(test_conv(:)),
            q = q(:,:,1:i+1);
            p = p(:,:,1:i+1);
            
            q_final = q(:,:,i+1);
            p_final = p(:,:,i+1);
            sp_final = sp_save(:,:,:,i+1);

            i,
            return
            
        elseif (i == num_iters-1) && (seg_num == size(q_init,2)),
            q_final = q(:,:,i+1);
            p_final = p(:,:,i+1);
            sp_final = sp_save(:,:,:,i+1);
            
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
    
end