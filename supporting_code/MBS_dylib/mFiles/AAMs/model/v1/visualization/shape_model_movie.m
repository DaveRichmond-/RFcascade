function [] = shape_model_movie(xbar, Psi, Lambda, p_axis, n_sigma, varargin)

% makes movie of statistical shape model
% Psi     : matrix of eigenvectors from PCA
% Lambda  : vector of eigenvalues from PCA
% p_axis  : principle axis to vary along
% n_sigma : number of stdev's to go along principle axis

if nargin >= 6,
    fname = varargin{1};
else
    fname = '';
end

numpoints = size(Psi,1)/2;

% set up movie object

fname = strcat(fname,'_shape variation along principle axis ',num2str(p_axis),'.avi');
writerObj = VideoWriter(fname);
open(writerObj);

% range of parameters to sweep over

lambda = Lambda(p_axis);
sigma_p = sqrt(lambda);
deltaS = 6*sigma_p/99;
range = [ [0 : deltaS : n_sigma*sigma_p], [n_sigma*sigma_p : -deltaS : -n_sigma*sigma_p], [-n_sigma*sigma_p : deltaS : 0] ];

% create figure and plot instances of model

fhandle = figure;

for i=1:length(range);
    
    x_1 = xbar + Psi(:,p_axis)*range(i);
    plot(x_1(1:numpoints), x_1(numpoints+1:2*numpoints),'o',...
        'LineWidth',2,...
        'MarkerEdgeColor','k',...
        'MarkerSize',10),
    hold on
    fnplt(cscvn([[x_1(1:numpoints);x_1(1)]';[x_1(numpoints+1:2*numpoints);x_1(numpoints+1)]']),...
        'r',2)
    text(-0.58,-0.6,strcat('b(1) = ',num2str(range(i))),'VerticalAlignment','bottom','HorizontalAlignment','left','FontSize',24)
    hold off
    
    axis([-0.6 0.6 -0.6 0.6]),
    set(gca,'XTick',[],'YTick',[]),
    
    frame = getframe(fhandle);
    writeVideo(writerObj,frame);
    
    
end

close(writerObj),
