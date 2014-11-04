function [] = plot_fit2AAM(shape, s_axis, appear, a_axis)

%

color_str = ['r','g','b','c','m','k'];

%

figure,
hold on,

for i = 1:length(shape),
    
    plot(shape(i).frames, shape(i).b(s_axis,:), color_str(mod(i-1,6)+1), ...
        'LineWidth',2)
    
end
xl = xlabel('Time [frame #]'); set(xl,'fontsize',20),
yl = ylabel(strcat('Weight b_',num2str(s_axis))); set(yl,'fontsize',20),
tit = title('Description of somite shape through time'); set(tit,'fontsize',24),

figure,
hold on,

for i = 1:length(appear),
    
    plot(appear(i).frames, appear(i).b(a_axis,:), color_str(mod(i-1,6)+1), ...
        'LineWidth',2)
    
end
xl = xlabel('Time [frame #]'); set(xl,'fontsize',20),
yl = ylabel(strcat('Weight b_',num2str(a_axis))); set(yl,'fontsize',20),
tit = title('Description of somite appearance through time'); set(tit,'fontsize',24),