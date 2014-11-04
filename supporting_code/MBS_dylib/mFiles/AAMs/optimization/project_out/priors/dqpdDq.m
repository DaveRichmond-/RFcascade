function [dScdDq1] = Prior_reparameterization(s0,Sj,Sja,weights,q,p,dq,dp);

% calculate position s1
s1 = s0;

for i = 1:size(Sj,3),
    
    s1 = s1 - dp(i)*Sj(:,:,i);
    
end

% calculate position s2
[st1, s2] = LK_Warp(s0,Sj,Sja,weights,-dq,-dp);

% calculate parameters of TSP Warp
[st2, s_temp] = LK_Warp(s0,Sj,Sja,weights,q,p);

% parse coefficients of TSP for simplicity 
a = st2.coefs(:,9:12);
W = st2.coefs(:,1:8);

% precompute
for k=1:8,
    for j=1:8,
        for i=1:2,
            
            WdUdDq1(i,j,k) = 2 * W(i,k) * [s1(1,j)*(s0(1,k)-s2(1,j)) + s1(2,j)*(s0(2,k)-s2(2,j))] * ...
                [1 + log( (s0(1,k)-s2(1,j))^2 + (s0(2,k)-s2(2,j))^2 )];

        end
    end
end

dScdDq1 = (1/weights(1))*( -a(:,1:2)*s1 + sum(WdUdDq1,3) );