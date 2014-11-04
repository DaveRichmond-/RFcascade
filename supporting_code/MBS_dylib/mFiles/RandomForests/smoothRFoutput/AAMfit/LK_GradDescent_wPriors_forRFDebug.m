function [q_final,p_final, sp_final, q, p, sp_save, conv_flag, cost] = LK_GradDescent_wPriors_forRFDebug(raw_image,modelSegmentsAAM,q_init,p_init,LKparams)

%

% unpack everything from modelSegmentsAAM
A0_vec = modelSegmentsAAM.A0_vec;
Grad_A0_vec = modelSegmentsAAM.Grad_A0_vec;
Ai_vec = modelSegmentsAAM.Ai_vec;
Grad_Ai_vec = modelSegmentsAAM.Grad_Ai_vec;
dNdq = modelSegmentsAAM.dNdq;
dWdp = modelSegmentsAAM.dWdp;
s0 = modelSegmentsAAM.s0;
s0_vec = modelSegmentsAAM.s0_vec;
Sj = modelSegmentsAAM.Sj;
Sj_vec = modelSegmentsAAM.Sj_vec;
Sja = modelSegmentsAAM.Sja;
Sja_vec = modelSegmentsAAM.Sja_vec;
weights = modelSegmentsAAM.weights;
SFP_positions = modelSegmentsAAM.SFP_positions;
pixelList = modelSegmentsAAM.pixelList;
H = modelSegmentsAAM.H;
SD = modelSegmentsAAM.SD;

% unpack LKparams
num_iters = LKparams.num_iters;
reg_weights = LKparams.reg_weights;
step_size = LKparams.step_size;
thresh = LKparams.conv_thresh;

% initialize
conv_flag = 0;
w_image_vec = NaN(size(A0_vec));
c = sqrt(reg_weights);      % simpler to use the sqrt(reg_weight) for the math... 
q_prior = q_init;           % THIS IS REDUNDANT.  WILL ALWAYS INITIATE AT THE SAME VALUE AS THE PRIOR.  DROP ONE OF THESE.
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

% grad descent loop
for i = 1:num_iters,

    %i
    % (1) Warp I with W(x;p) followed by N(x;q); where p,q come from last iteration ---------------->
    
    % calculate TSP warp from movement of base-mesh points under p, q
    [st1, s1] = LK_Warp(s0,Sj,Sja,weights,q(:,i),p(:,i));
    
    w_positions = fnval(st1, SFP_positions);
    
    if sum((w_positions(1,:) < 1) + (w_positions(2,:) < 1) + (w_positions(1,:) > size(raw_image,1)) + (w_positions(2,:) > size(raw_image,2))) > 0,
        q = q(:,1:i-1);
        p = p(:,1:i-1);
        
        q_final = q(:,i-1);
        p_final = p(:,i-1);
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
        w_image_vec_MEAN = mean(w_image_vec);
        w_image_vec = w_image_vec - w_image_vec_MEAN;
        w_image_vec_NORM = norm(w_image_vec);
        w_image_vec = w_image_vec / w_image_vec_NORM;
    else
        w_image_vec = w_image_vec - w_image_vec_MEAN;
        w_image_vec = w_image_vec / w_image_vec_NORM;
    end
    
    % (2) Compute the error image
    Err_image_vec = w_image_vec - A0_vec;
    
    % <NEW>
    
    if i==1,
        
        inv_H = H^-1;
        SD_update = SD' * Err_image_vec;
        
    else,
        
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
        dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q(:,i-1),p(:,i-1),dq(:,i-1),dp(:,i-1),qp(:,i-1),pp(:,i-1));
        
        % (6a) calc Hessian and invert
        H_prior = (dFdp'*dppdDp)'*(dFdp'*dppdDp);
        inv_H = (H + H_prior)^-1;
        
        % (7) Compute SD update
        SD_update = SD' * Err_image_vec;
        SD_update = SD_update - (dFdp'*dppdDp)'*F;
        
    end
    
    % </NEW>
    
    % (8) Compute the parameter updates: (dq,dp)
    Delta = inv_H * SD_update;                                   % MINUS SIGN MISSING ???
    dq(:,i) = Delta(1:4);
    dp(:,i) = Delta(5:end);
    
    % shrink the step size to avoid oscillations                        % CAREFUL !!!
    dq(:,i) = dq(:,i).*step_size(1:4);
    dp(:,i) = dp(:,i).*step_size(5:end);
    
    % (9) Update parameters by (dq,dp)
    [qp(:,i), pp(:,i), sp] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q(:,i),p(:,i),dq(:,i),dp(:,i));
    
    % re-assign for next iteration
    q(:,i+1) = qp(:,i);
    p(:,i+1) = pp(:,i);
    
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
    
    test_conv = abs([diff_q; diff_p]) < thresh;
    
    if prod(test_conv),
        q = q(:,1:i+1);
        p = p(:,1:i+1);
        
        q_final = q(:,i+1);
        p_final = p(:,i+1);
        sp_final = sp;

        i,
        return
    elseif i == num_iters-1,
        q_final = q(:,i+1);
        p_final = p(:,i+1);
        sp_final = sp;
        
        conv_flag = 2;
        display(strcat('doesnt converge in ',num2str(num_iters-1),' iterations')),
        return
    end
    
    if i == 1,
        cost(i) = sum(Err_image_vec.^2);
    else
        cost(i) = sum(Err_image_vec.^2) + F^2;
    end
    
    % cost based convergence test
    %{
    
    
    if (i ~= 1) && ((cost(i-1) - cost(i)) < thresh),
        
        i
        q_final = qp(:,i);
        p_final = pp(:,i);
        sp_final = sp;
        return
        
    end
    %}
    
end