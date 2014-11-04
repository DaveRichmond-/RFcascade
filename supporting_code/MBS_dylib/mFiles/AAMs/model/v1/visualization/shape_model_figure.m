function [] = shape_model_figure(xbar, Psi, Lambda, p_axis, b_val)

numpoints = length(xbar)/2;

fhandle = figure;

x_1 = xbar + Psi(:,p_axis)*b_val;
plot(x_1(1:numpoints), x_1(numpoints+1:2*numpoints),'o',...
    'LineWidth',2,...
    'MarkerEdgeColor','k',...
    'MarkerSize',10),
hold on
% fnplt(cscvn([[x_1(1:numpoints);x_1(1)]';[x_1(numpoints+1:2*numpoints);x_1(numpoints+1)]']),...
%     'r',2)
text(-0.58,-0.6,strcat('b(1) =',' ',' ',num2str(b_val)),'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',24)
hold off

axis([-0.6 0.6 -0.6 0.6]),
set(gca,'XTick',[],'YTick',[]),