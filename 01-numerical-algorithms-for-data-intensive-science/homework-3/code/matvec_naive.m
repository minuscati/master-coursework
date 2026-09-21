function b = matvec_naive(x, A)
n = size(A,1);
b=zeros(n,1);
for i=1:n
    for j=1:n
        b(i)=b(i)+A(i,j)*x(j);
    end
end

end
