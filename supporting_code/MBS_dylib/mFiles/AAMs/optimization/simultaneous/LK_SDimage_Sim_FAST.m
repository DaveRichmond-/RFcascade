function [SD] = LK_SDimage_Sim_FAST(Grad_A0, dNdq, dWdp, Ai, L, Grad_Ai);

% 

% initialize
SD = NaN(size(dNdq,1), size(dNdq,3)+size(dWdp,3));

% merge contributions from eigen images - FASTER
sum_Grad_Ai = zeros(size(Grad_Ai,1),size(Grad_Ai,2));
for i = 1:size(Grad_Ai,3),
    sum_Grad_Ai = sum_Grad_Ai + L(i)*Grad_Ai(:,:,i);
end
clear i

% calc SD w.r.t warp Jacobians
all_Grad = Grad_A0 + sum_Grad_Ai;

%
for j = 1:size(SD,2),
    if j <= 4,
        SD(:,j) = sum(all_Grad.*dNdq(:,:,j),2);
    else
        SD(:,j) = sum(all_Grad.*dWdp(:,:,j-4),2);
    end
end

% append Ai to SD
SD = [SD, Ai];