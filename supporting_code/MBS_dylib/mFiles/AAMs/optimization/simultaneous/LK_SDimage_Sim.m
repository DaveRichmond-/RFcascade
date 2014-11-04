function [SD] = LK_SDimage_Sim(Grad_A0, dNdq, dWdp, Ai, L, Grad_Ai);

% 

% initialize
SD = NaN(size(dNdq,1), size(dNdq,3)+size(dWdp,3));

% merge contributions from eigen images
sum_Grad_Ai = zeros(size(Grad_Ai,1),size(Grad_Ai,2));
for i = 1:size(Grad_Ai,3),
    sum_Grad_Ai = sum_Grad_Ai + L(i)*Grad_Ai(:,:,i);
end
clear i

% compute standard SD images
for i = 1:size(SD,1),
    
    for j = 1:size(SD,2),
        
        if j <= 4,
            
            SD(i,j) = dot(Grad_A0(i,:)+sum_Grad_Ai(i,:), dNdq(i,:,j));
            
        else
            
            SD(i,j) = dot(Grad_A0(i,:)+sum_Grad_Ai(i,:), dWdp(i,:,j-4));
            
        end
        
    end
    
end

SD = [SD, Ai];