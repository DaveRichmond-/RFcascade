function [sp_final, cost] = LK_GradDescent_Sim_wPriors_forRF(raw_image, modelSegmentsAAM, q_init, p_init, L_init, LKparams)

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
% pixelList = modelSegmentsAAM.pixelList;
    
% unpack LKparams
num_iters = LKparams.num_iters;
reg_weights = LKparams.reg_weights;
step_size = LKparams.step_size;
thresh = LKparams.conv_thresh;

% set priors to be initial values of parameters
q_prior = q_init;
p_prior = p_init;
L_prior = L_init;

% initialize
c = sqrt(reg_weights(:));      % simpler to use the sqrt(reg_weight) for the math...

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

%
dppdDp = zeros(length(q_init)+length(p_init)+length(L_init));

% grad descent loop
for i = 1:num_iters,

    % (1) Warp I with W(x;p) followed by N(x;q); where p,q come from last iteration ---------------->
    
    % calculate TSP warp from movement of base-mesh points under p, q
    [st1, s1] = LK_Warp(s0,Sj,Sja,weights,q(:,i),p(:,i));
    
    w_positions = fnval(st1, SFP_positions);
    
    if sum((w_positions(2,:) < 1) + (w_positions(1,:) < 1) + (w_positions(2,:) > size(raw_image,1)) + (w_positions(1,:) > size(raw_image,2))) > 0,
        
        if i == 1,
            
            [qp(:,i), pp(:,i), sp_final] = LK_ParamsOf_Comp_Warps(s0,Sj,Sja,weights,q(:,i),p(:,i),zeros(size(q,1),1),zeros(size(p,1),1));
            cost = 1/eps;
            
        else
            
            sp_final = sp;        
            cost = 1/eps;
            
        end
        
        display('template has left the image domain')
        return
        
    end
    
    w_image_vec = bilinear_interp_vec(raw_image, w_positions(2,:), w_positions(1,:));

    % normalize w_image_vec with respect to first guess of position
    % (INPUT IMAGE IS ALREADY NORMALIZED)
    %{
    if i==1,
        w_image_vec_MEAN = mean(w_image_vec); %repmat(mean(w_image_vec), [size(w_image_vec,1),1]);
        w_image_vec_STD = std(w_image_vec); %repmat(sqrt(sum(w_image_vec.^2,1)), [size(w_image_vec,1),1]);
    end
    w_image_vec = w_image_vec - w_image_vec_MEAN;
    w_image_vec = w_image_vec / w_image_vec_STD;
    %}
    
    % (2) Compute the error image
    Err_image_vec =  A0_vec + Ai_vec*L(:,i) - w_image_vec;         % note the change of sign from before!
    
    % compute the SD image
    SD = LK_SDimage_Sim_FAST(Grad_A0_vec, dNdq, dWdp, Ai_vec, L(:,i), Grad_Ai_vec);
    
    % compute the Hessian
    H = LK_Hessian(SD);
    
    % <NEW>
    
    if i==1,
        
        inv_H = H^-1;
        SD_update = SD' * Err_image_vec;
        
    else
        
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
        dppdDp(1:length(q_init)+length(p_init),1:length(q_init)+length(p_init)) = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q(:,i-1),p(:,i-1),dq(:,i-1),dp(:,i-1),qp(:,i-1),pp(:,i-1));
        dppdDp(1:length(q_init)+length(p_init), length(q_init)+length(p_init)+1:end) = zeros(length(q_init)+length(p_init),length(L_prior));
        dppdDp(length(q_init)+length(p_init)+1:end, 1:length(q_init)+length(p_init)) = zeros(length(L_prior),length(q_init)+length(p_init));
        dppdDp(length(q_init)+length(p_init)+1:end, length(q_init)+length(p_init)+1:end) = eye(length(L_prior));        
%         dppdDp = prior_reparameterization(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,q(:,i-1),p(:,i-1),dq(:,i-1),dp(:,i-1),qp(:,i-1),pp(:,i-1));
%         dppdDp = [dppdDp, zeros(size(dppdDp,1),length(L_prior)); zeros(length(L_prior),size(dppdDp,2)), eye(length(L_prior))];
        
        % (6a) calc Hessian and invert
        H_prior = (dGdp'*dppdDp)'*(dGdp'*dppdDp);
        inv_H = (H + H_prior)^-1;

        % (7) Compute SD update
        SD_update = SD' * Err_image_vec + (dGdp'*dppdDp)'*G;        % MINUS SIGNS CANCEL ON PRIOR TERM
        
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
    
    % parameter based convergence test
    diff_q = q(:,i+1) - q(:,i);
    diff_p = p(:,i+1) - p(:,i);
    diff_L = L(:,i+1) - L(:,i);
    
    test_conv = abs([diff_q; diff_p; diff_L]) < thresh;
    
    if prod(test_conv),
        
        sp_final = sp;

        if i == 1,
            cost = sum(Err_image_vec.^2);
        else
            cost = sum(Err_image_vec.^2) + G^2;
        end

        return
        
    elseif i == num_iters,
        
        sp_final = sp;
        
        if i == 1,
            cost = sum(Err_image_vec.^2);
        else
            cost = sum(Err_image_vec.^2) + G^2;
        end
        
        return
        
    end
    
end