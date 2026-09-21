function Z=load_mat_hw1(n,p)
   randn('seed',0)
   p1=floor(3*p/4)
   p2=p-p1;
   Q=randn(n,p1);
   Z1=Q*diag(randn(1,p1).*exp(-abs(1:p1)))*randn(p1,p1);
   [~,I]=sort(sum(abs(Z1),1))
   Z1=Z1(:,I);
   Q=randn(n,p2);
   Z2=Q*diag(randn(1,p2).*exp(-abs(1:p2)))*randn(p2,p2);   
   Z=[Z1,Z2];
end

