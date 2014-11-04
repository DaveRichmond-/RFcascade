function [] = shape_fit_movie(som_num, xbar, Psi, b_vals, frames)

%

numpoints = length(xbar)/2;

%

fname = strcat('Fit som#',num2str(som_num),' to shape model.avi');
writerObj = VideoWriter(fname);
open(writerObj);

fhandle = figure('color','black');

for i=1:length(frames);
    
    x_1 = xbar + Psi(:,1)*b_vals(i);
    plot(x_1(1:numpoints), x_1(numpoints+1:2*numpoints),'o',...
        'LineWidth',2,...
        'MarkerEdgeColor','w',...
        'MarkerSize',10),
    hold on
    fnplt(cscvn([[x_1(1:numpoints);x_1(1)]';[x_1(numpoints+1:2*numpoints);x_1(numpoints+1)]']),...
        'w',2)
    %text(-0.58,-0.6,strcat('b(1) = ',num2str(b_vals(i))),'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',24)
    hold off
    
    axis([-0.6 0.6 -0.6 0.6]),
    set(gca,'XTick',[],'YTick',[]),
    set(gca, 'Color', 'k')
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
        
end

close(writerObj),