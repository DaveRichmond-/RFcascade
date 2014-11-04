function [q_init p_init] = update_priors(t, q_init_save, p_init_save, conv_flags, prior_model, hyst);

if t <= hyst, %sum(conv_flags == 0) < hyst,
    
    for i=1:4,
        q_init(i) = polyval(prior_model(i,:), t);
    end
    p_init = polyval(prior_model(5,:), t);
   
else
    
    for i=1:4,
        P = polyfit([t-hyst : t-1], q_init_save(i,[t-hyst : t-1]), 1);
        q_init(i) = polyval(P, t);
    end
    P = polyfit([t-hyst : t-1], p_init_save(1,[t-hyst : t-1]), 1);
    p_init = polyval(P, t);
    
end

q_init = q_init(:);