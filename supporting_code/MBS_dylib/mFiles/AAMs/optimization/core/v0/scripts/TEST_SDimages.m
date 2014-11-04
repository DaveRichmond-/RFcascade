%% test SDimages

[SD_mod] = LK_SDimage(Grad_A0, dNdq, dWdp, Ai_vec);

SD_images = NaN(size(X,1),size(X,2),size(SD_mod,2));

for i = 1:size(SD_images,3),
    
    temp = SD_images(:,:,i);
    temp(SFP_indx) = SD_mod(:,i);
    SD_images(:,:,i) = temp;
    
end