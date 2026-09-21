function b = matvec_hodlr(x, A, v, index_dict, child_dict)

n = size(A,1);
b=zeros(n,1);

node_keys=keys(index_dict);

for tau=1:length(node_keys);

    node_key=node_keys(tau);
    Itau=index_dict(node_key{1});
    if (~isKey(child_dict,node_key))
        b(Itau)=b(Itau)+A(Itau, Itau)*x(Itau); % What goes here?
    else
        children=child_dict(node_key{1});

        child1=children(1,:);
        child2=children(2,:);
        Isigma1=index_dict(child1);
        Isigma2=index_dict(child2);

        % block1_add=A(Isigma1,Isigma2)*x(Isigma2); % What goes here?
        block1_add=v(Isigma1) * (v(Isigma2)'*x(Isigma2));
        % block2_add=A(Isigma2,Isigma1)*x(Isigma1); % What goes here?
        block2_add=v(Isigma2) * (v(Isigma1)'*x(Isigma1));
        b(Itau)=b(Itau)+[block1_add;block2_add];
    end
end


end