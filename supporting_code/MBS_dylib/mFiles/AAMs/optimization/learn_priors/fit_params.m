function [params] = fit_params(som_nums);

%

num_p = 16;
num_lambda = 535;

fname_model = 'som7_model.mat';
fname_image = 'Gblur_r=2.tif';
last_frame = 144;

[shape,appear,s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,A0,A0_vec,Ai,Ai_vec,Grad_A0_vec,Grad_Ai_vec,dWdp,dNdq,SFP_positions,pixelList,Trev_imageStack] = LK_precompute_Simultaneous(fname_model,fname_image,last_frame,num_p,num_lambda);

% turn Trev_imageStack back to Tforward
imageStack = flipdim(Trev_imageStack,3);

%
load('fullDataSet.mat'),

for i=1:length(som_nums),
    
    i
    
    som_indx = find(dataSet(:,7) == som_nums(i));
    
    % read in frames
    all_frames = unique(dataSet(som_indx,4));
    all_frames = sort(all_frames,'ascend');
    
    for k = 1:length(all_frames),
        
        raw_image = imageStack(:,:,all_frames(k));
        
        frame_indx = find(dataSet(som_indx,4) == all_frames(k));

        s_pos = dataSet(som_indx(frame_indx), 2:3)';
        s_pos_vec = reshape(permute(s_pos,[2 1]),[prod(size(s_pos)),1]);
        
        for j = 1:size(Sja,3),
            q(j) = dot(Sja_vec(:,j), (s_pos_vec - s0_vec));
        end
        
        [A_temp, b_temp] = LK_qtoA(q, weights);
        
        s_unq = (A_temp^-1)*(s_pos - repmat(b_temp,[1,size(s0,2)]));
        s_unq_vec = reshape(permute(s_unq,[2 1]),[prod(size(s_unq)) 1]);
        
        % back-calculate p's
        for j = 1:size(Sj,3),
            p(j) = dot(Sj_vec(:,j), (s_unq_vec - s0_vec));
        end
        
        % calculate p
        %[p err] = fit_shape_model(s_unq_vec, s0_vec, Sj_vec);
        
        % calc lambda
        [st1, s1] = LK_Warp(s0,Sj,Sja,weights,q,p);
    
        w_positions = fnval(st1, SFP_positions);
        
        for m = 1:length(pixelList),
            w_image_vec(m) = bilinear_interp(raw_image, w_positions(2,m), w_positions(1,m));
        end
        w_image_vec = w_image_vec(:);
        
        % normalize w_image_vec
        w_image_vec_MEAN = repmat(mean(w_image_vec), [size(w_image_vec,1),1]);
        w_image_vec_STD  = repmat(sqrt(sum(w_image_vec.^2,1)), [size(w_image_vec,1),1]);
        w_image_vec = (w_image_vec - w_image_vec_MEAN)./ w_image_vec_STD;

        %[L_FIM(:,k), err] = fit_appearance_model(w_image_vec, A0_vec, Ai_vec);

        % (2) Compute the error image
        Err_image_vec = w_image_vec(:) - A0_vec;
        
        %{
        figure,
        imagesc(raw_image),
        colormap('gray'),
        axis([231 392 150 550]),
        hold on,
        set(gca,'XTick',[],'YTick',[]),
        
        hold on,
        
        plot(s1(1,:),s1(2,:),'wo')
        %}
        
        for j = 1:size(Ai_vec,2),
            L(j) = dot(Ai_vec(:,j), Err_image_vec);
        end
        
        %store
        params(i).frame(k) = all_frames(k);
        params(i).q(:,k) = q(:);
        params(i).p(:,k) = p(:);
        params(i).L(:,k) = L(:);
        
        clear frame_indx s_pos s_pos_vec A_temp b_temp s_unq s_unq_vec
    end
    
end