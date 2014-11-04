%% script to build "model backbone" with instances of AAM, to be used for initializing AAMs in RF cascade

%%

num_p = 1;
num_lambda = 1;

fname_model = 'som7_model.mat';
fname_image = 'label_registered22.tif';
last_frame = 1;

[shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,Ai,Ai_vec,Grad_A0_vec,Grad_Ai_vec,dWdp,dNdq,SFP_positions,pixelList,Trev_imageStack] = LK_precompute_Simultaneous(fname_model,fname_image,last_frame,num_p,num_lambda);

clear Trev_imageStack

%% initiate warp

% select somite to track, and last frame (to start tracking from)
som_nums = [1:21];

for i=1:length(som_nums),
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    s_pos = dataSet(som_indx, 2:3)';
    s_pos_vec = reshape(permute(s_pos,[2 1]),[prod(size(s_pos)),1]);
    
    for j = 1:size(Sja,3),
        q_init(j,i) = dot(Sja_vec(:,j), (s_pos_vec - s0_vec));
    end

    [A_temp,b_temp] = LK_qtoA(q_init(:,i), weights);
    
    s_unq = (A_temp^-1)*(s_pos - repmat(b_temp,[1,size(s0,2)]));
    s_unq_vec = reshape(permute(s_unq,[2 1]),[prod(size(s_unq)) 1]);
    
    % back-calculate pp's
    
    for j = 1:size(Sj,3),
        
        p_init(j,i) = dot(Sj_vec(:,j), (s_unq_vec - s0_vec));
        
    end
    
    clear som_indx frame_inx s_pos A_temp b_temp s_unq
    
end

L_init = 0;

%%

