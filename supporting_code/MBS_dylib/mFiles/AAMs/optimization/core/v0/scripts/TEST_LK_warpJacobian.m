%% script to explore dWdp for Thin Plate Splines and AAMs

%% run warpJacobian.m to calc dWdp and dNdq

% define characteristic matrices from model

s0 = reshape(shape(1).xbar_vector, [8,2])';
Sj = permute(reshape(shape(1).Psi, [8,2,16]), [2 1 3]);

% make corresponding e-vectors of Similarity Transform

R = [0 -1; 1 0];

Sja(:,:,1) = s0;
Sja(:,:,2) = R * s0;
Sja(:,:,3) = [ones(1,size(s0,2)); zeros(1,size(s0,2))];
Sja(:,:,4) = [zeros(1,size(s0,2)); ones(1,size(s0,2))];

% run

[dWdp, dNdq] = warpJacobian(X,Y,pixelList,s0,Sj,Sja);

%% raster dWdp into real images

im = NaN(size(X,1),size(X,2),size(dWdp,2),size(dNdq,3)+size(dWdp,3));

for i=1:size(dWdp,2),
    
    for j=1:size(dNdq,3),

        im(:,:,i,j) = make_image_from_image_vector(dNdq(:,i,j),X,pixelList);
        
    end
    
    for j=1:size(dWdp,3),
        
        im(:,:,i,4+j) = make_image_from_image_vector(dWdp(:,i,j),X,pixelList);
        
    end
    
end
        
%% plot

Sj_all = zeros(size(Sj,1),size(Sj,2),size(Sja,3)+size(Sj,3));
Sj_all(:,:,1:size(Sja,3)) = Sja;
Sj_all(:,:,size(Sja,3)+1:end) = Sj;

for j = 1:4,%size(im,4),

    figure,
    
    for i = 1:size(im,3),
    
        %subplot(size(im,3),size(im,4),(i-1)*size(im,4) + j),
        subplot(2,1,i),
        imagesc(X(1,:),Y(:,1),im(:,:,i,j)),
        set(gca,'XTick',[],'YTick',[])
        colorbar,
        hold on,
        
        plot(s0(1,:),s0(2,:),'o',...
            'LineWidth',2,...
            'MarkerEdgeColor','w',...
            'MarkerSize',10),
        
        if j<=4,
            
            head = s0 + 0.25*Sj_all(:,:,j);
            
        else
            
            head = s0 + 3*sqrt(Lambda(j-4,j-4))*Sj_all(:,:,j);
    
        end
        
        for k = 1:size(s0,2),

            if i == 1,
                
                line([s0(1,k) head(1,k)],[s0(2,k) s0(2,k)],'Color','w','LineWidth',2),
                
            elseif i == 2,
                
                line([s0(1,k) s0(1,k)],[s0(2,k) head(2,k)],'Color','w','LineWidth',2),
                
            end
            
        end
        
        axis(0.6*[-1 1 -1 1])
            
    end
    
end

% plot wrt theta and k
%{
dq1du1 = X(pixelList)./sqrt(X(pixelList).^2 + Y(pixelList).^2);
dq2du1 = Y(pixelList)./sqrt(X(pixelList).^2 + Y(pixelList).^2);

dNdu(:,1,1) = dNdq(:,1,1).*dq1du1 + dNdq(:,1,2).*dq2du1;
dNdu(:,2,1) = dNdq(:,2,1).*dq1du1 + dNdq(:,2,2).*dq2du1;

im_test(:,:,1) = make_image_from_image_vector(dNdu(:,1,1),X,pixelList);
im_test(:,:,2) = make_image_from_image_vector(dNdu(:,2,1),X,pixelList);

figure,
subplot(2,1,1)
imagesc(X(1,:),Y(:,1),im_test(:,:,1)),
colorbar
subplot(2,1,2)
imagesc(X(1,:),Y(:,1),im_test(:,:,2)),
colorbar
%}