function [q_new,p_new,sp] = iterate_LK(s0,Sj,Sja,weights,SFP_positions,pixelList,raw_image,A0_vec,SD,inv_H,q_init,p_init,thresh);

% run steps (1,2,7,8,9) from Lucas-Kanade algorithm
% 

q_old = q_init;
p_old = p_init;

for i = 1:100,
    
    % (1) Warp I with W(x;p) followed by N(x;q); where p,q come from last iteration ---------------->
    
    % calculate TSP warp from movement of base-mesh points under p_old, q_old
    [st1, s1] = LK_Warp(s0,Sj,Sja,weights,q_old,p_old);
    
    % store shape during grad descent
    if i == 1,
        sp_save(:,:,i) = s1;
    end
    
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
    Err_image_vec = w_image_vec' - A0_vec;
    
    % (7) Compute SD update
    SD_update = SD' * Err_image_vec;
    
    % (8) Compute the parameter updates: (dq,dp)
    Delta_qp = inv_H * SD_update;
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
    %{
    diff_q = q_new' - q_old;
    diff_p = p_new' - p_old;
    
    store_diff_qp(:,i) = [diff_q; diff_p];
    %}
    
    % test convergence
    diff_q = abs(q_new' - q_old);
    diff_p = abs(p_new' - p_old);
    
    % re-assign for next iteration
    q_old = q_new';
    p_old = p_new';
    
    % store position
    sp_save(:,:,i+1) = sp;
    cost(i) = sum(Err_image_vec.^2);
    
    if (i ~= 1) && (cost(i) - cost(i-1)) < thresh,
        break
    end
    
end