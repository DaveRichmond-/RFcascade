%%

num_classes = 22;
num_levels = 3;
num_images = 10;

load('diceScores.txt')

diceScores = reshape(diceScores, [num_classes,num_levels,num_images]);
diceScores = permute(diceScores, [2,1,3]);

%%

classIdx = [0:num_classes-1];

for i = 1:size(diceScores,3)
    figure
    bar(classIdx, permute(diceScores(:,:,i),[2,1,3]))
    axis([-1 classIdx(end)+1 0 1])
    legend('first level','second level','third level','fourth level','fifth level')
end


% bar(repmat(classIdx, [1, 3]),[diceScores(1,:,1);diceScores(2,:,1);diceScores(3,:,1)],'grouped')