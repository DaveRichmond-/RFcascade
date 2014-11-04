%% script to run test_position_priors

c = [1e3];

q_init = [1;1;1;1];
p_init = 0.1;

dq_init = [0.1;0.1;0.1;0.1];
dp_init = 0.05;

q_offset = q_init;
p_offset = p_init;

[st, s_init] = LK_Warp(s0,Sj,Sja,weights,q_init,p_init);

[st, s_const] = LK_Warp(s0,Sj,Sja,weights,q_init + q_offset,p_init + p_offset);

pos_nums = [3 4 5];

[q,p] = test_position_priors(s0,s0_vec,Sj,Sj_vec,Sja,Sja_vec,weights,c,q_init,p_init,dq_init,dp_init,pos_nums,s_const(:,pos_nums));

%% show distance to rc

for i=1:size(q,2),
    
    [st, s_new] = LK_Warp(s0,Sj,Sja,weights,q(:,i),p(:,i));
    temp = (s_new - s_const).^2;
    dist(i) = sum(sum(temp));
    
end

figure,
plot(dist)

%% plot convergence


save_flag = 1;
fname = 'test convergence with position priors_erase.avi';

if save_flag,
    
    writerObj = VideoWriter(fname);
    open(writerObj);

end

set(0,'DefaultFigureWindowStyle','normal'),
fhandle = figure;

for i=1:size(q,2),
    
    plot(s_const(1,:),s_const(2,:),'ko'),
    axis([-2 2 -2 2])
    hold on,
    plot(s_const(1,pos_nums),s_const(2,pos_nums),'go'),
    
    
    [st, s_new] = LK_Warp(s0,Sj,Sja,weights,q(:,i),p(:,i));
    plot(s_new(1,:),s_new(2,:),'ro')
    plot(s_new(1,pos_nums),s_new(2,pos_nums),'go'),
    
    if 1-save_flag,
        pause(0.25),
    end
    
    if save_flag,
        frame = getframe(fhandle);
        writeVideo(writerObj,frame);
    end
    clf
    
end

if save_flag,
    close(writerObj),
end

set(0,'DefaultFigureWindowStyle','docked'),