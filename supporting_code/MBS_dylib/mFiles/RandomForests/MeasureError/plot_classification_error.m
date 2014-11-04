%% script to plot classification error

labels = {'rf1','rf3','rf6','rf10','rf30','rf100'};
min_split_criterion = [1 3 6 10 30 100];
classification_error = [];  % transfer manually
oob_error = [];  % transfer manually

%%

set(0,'DefaultFigureWindowStyle','docked'),

figure, hold on,

plot(min_split_criterion, oob_error)
plot(min_split_criterion, mean(classification_error,1),'r')
plot(min_split_criterion, oob_error,'bo')
plot(min_split_criterion, mean(classification_error,1),'ro')

leg = legend('oob error','classification error');
set(leg,'fontsize',14);