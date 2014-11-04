function [params] = fit_params();

%

num_p = 1;
num_lambda = 1;

fname_model = 'som7_model.mat';
fname_image = 'f1-58_blur.tif';
last_frame = 58;

[shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,Ai,Ai_vec,Grad_A0_vec,Grad_Ai_vec,dWdp,dNdq,SFP_positions,pixelList,Trev_imageStack] = LK_precompute_Simultaneous(fname_model,fname_image,last_frame,num_p,num_lambda);


%

%
load('fullDataSet.mat'),

% select somite to track, and last frame (to start tracking from)
som_nums = [7];

for i=1:length(som_nums),
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    
    % read in frames
    all_frames = unique(dataSet(som_indx,4));
    
    for k = 1:length(all_frames),
        
        frame_indx = find(dataSet(som_indx,4) == all_frames(k));

        s_pos = dataSet(som_indx(frame_indx), 2:3)';
        s_pos_vec = reshape(permute(s_pos,[2 1]),[prod(size(s_pos)),1]);
        
        for j = 1:size(Sja,3),
            q_init(j) = dot(Sja_vec(:,j), (s_pos_vec - s0_vec));
        end
        
        [A_temp, b_temp] = LK_qtoA(q_init, weights);
        
        s_unq = (A_temp^-1)*(s_pos - repmat(b_temp,[1,size(s0,2)]));
        s_unq_vec = reshape(permute(s_unq,[2 1]),[prod(size(s_unq)) 1]);
        
        % back-calculate pp's
        
        for j = 1:size(Sj,3),
            
            p_init(j) = dot(Sj_vec(:,j), (s_unq_vec - s0_vec));
            
        end
        
        %store
        params(i).q(:,k) = q_init(:);
        params(i).p(:,k) = p_init(:);
        params(i).frame(k) = all_frames(k);
        
        clear som_indx frame_inx s_pos A_temp b_temp s_unq
    end
    
end