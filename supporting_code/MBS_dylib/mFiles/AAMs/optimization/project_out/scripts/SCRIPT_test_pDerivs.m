[0 -1; 1 0]*s0 - s0
%%
for i = 1:4,

    for j = 1:4,
        
        dqdDq2_Sj(i,j) = dot(reshape(permute([0 -1; 1 0]*Sj(:,:,i),[2 1]),[prod(size(s0)) 1]),Sja_vec(:,j));
        
    end
    
    dqdDq2_s0(i) = dot(reshape(permute([0 -1; 1 0]*s0 - s0, [2 1]),[prod(size(s0)) 1]),Sja_vec(:,i));
    
end

%%
for i = 1:4,
    
    dqdDq3_s0(i) = dot(Sja_vec(:,i),s0_vec);
    dqdDq3_Sc(i) = dot(Sja_vec(:,i),[ones(8,1); zeros(8,1)]);
    
end