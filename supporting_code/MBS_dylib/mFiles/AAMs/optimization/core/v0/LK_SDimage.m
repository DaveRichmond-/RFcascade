function [SD_mod] = LK_SDimage(Grad_A0, dNdq, dWdp, Ai_vec);

% 

% initialize

SD = NaN(size(dNdq,1), size(dNdq,3)+size(dWdp,3));
SD_mod = NaN(size(dNdq,1), size(dNdq,3)+size(dWdp,3));

% compute standard SD images

for i = 1:size(SD,1),
    
    for j = 1:size(SD,2),
        
        if j <= 4,
            
            SD(i,j) = dot(Grad_A0(i,:), dNdq(i,:,j));
            
        else
            
            SD(i,j) = dot(Grad_A0(i,:), dWdp(i,:,j-4));
            
        end
        
    end
    
end

corr = (((Ai_vec'*SD)')*Ai_vec')';

SD_mod = SD - corr;