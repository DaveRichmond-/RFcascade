function [shape, appear] = leaveOneOutFit(imageStack, dataSet);

% fit data set to itself using "leave one out" approach
%
%
%
%

% pre-calculate a few useful things

numpoints = size(unique(dataSet(:,1)), 1);               % 
som_nums = unique(dataSet(:,7));

%

for i = 1:length(som_nums),
    
    % [1] partition date into data for building model, and data to be fit to model ------------------------>
    
    model_data = [];
    
    for j = 1:length(som_nums),
        
        if som_nums(j) ~= som_nums(i),
            
            model_indx = find(dataSet(:,7) == som_nums(j));
            model_data = [model_data; dataSet(model_indx, :)];
        
        else
            
            somite_indx = find(dataSet(:,7) == som_nums(j));
            somite_data = dataSet(somite_indx,:);
            
        end
        
    end
    
    % [2] shape ---------------------------------->
    
    % make shape model from model_data
    
    [xbar_vector, R, Psi, Lambda, PsiT] = make_shape_model(model_data);
    xbar_mat = [xbar_vector(1:numpoints), xbar_vector(numpoints+1:2*numpoints)];
    
    % fit somite_data to shape model
    
    frames = unique(somite_data(:,4));
    
    for j=1:length(frames),
        
        % create shape vector
        
        indx = find(somite_data(:,4) == frames(j));
        xmat = somite_data(indx,2:3);
        x_vector = xmat(:);
                
        % fit to model
        
        [b(:,j) err(:,j)] = fit_shape_model(x_vector, xbar_vector, Psi);
        
    end
    
    % store everything
        
    shape(i).xbar_vector = xbar_vector;
    shape(i).R = R;
    shape(i).Psi = Psi;
    shape(i).Lambda = Lambda;
    shape(i).PsiT = PsiT;
    %shape(i).x_vector = x_vector;
    shape(i).frames = frames;
    shape(i).b = b;
    shape(i).err = err;

    % clear
    
    clear R Psi Lambda PsiT b err
    
    % [3] appearance --------------------------->

    % build model
    
    [gbar, R, Psi, Lambda, PsiT, X, Y, CC] = make_appearance_model(imageStack, model_data, xbar_vector);
    
    % calculate grid to populate with image vectors during fitting (next)
    
    sampled_positions = [X(CC.PixelIdxList{1,1}) Y(CC.PixelIdxList{1,1})];
    
    % fit appearance data to appearance model
    
    for j=1:length(frames),
        
        % create image vector
        
        indx = find(somite_data(:,4) == frames(j));
        shape_mat = somite_data(indx,2:3);
        
        % calculate warp function using Matlab's thin plate spline
        
        st = tpaps(xbar_mat', shape_mat', 1);
        warped_positions = fnval(st, sampled_positions')';
        
        % map image at warped_positions back into warped_image
        
        raw_image = imageStack(:,:,frames(j));
        warped_image = zeros(size(X,1), size(X,2));
        
        for m = 1:length(CC.PixelIdxList{1,1}),
            
            warped_image(CC.PixelIdxList{1,1}(m)) = bilinear_interp(raw_image, warped_positions(m,2),...
            warped_positions(m,1));
        
        end
        
        % write to image vector
        
        warped_image_vector = warped_image(CC.PixelIdxList{1,1});
        g(:,j) = warped_image_vector(:);
                
        % fit image vector to model
        
        [b(:,j) err(:,j)] = fit_appearance_model(g(:,j), gbar, Psi);
        
    end
    
    % store everything
    
    appear(i).gbar = gbar;
    appear(i).R = R;
    appear(i).Psi = Psi;
    appear(i).Lambda = Lambda;
    appear(i).PsiT = PsiT;
    appear(i).X = X;
    appear(i).Y = Y;
    appear(i).CC = CC;
    appear(i).frames = frames;
    appear(i).b = b;
    appear(i).err = err;

    % clear everything
    
    clear xbar_vector gbar R Psi Lambda PsiT X Y CC b err
    
end 
