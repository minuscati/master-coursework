function [L,V,E]=Simi_graph(A,c,ee)
    % Distance Marix
    Dist=zeros(size(A,1));
    for i=1:size(A,1)
       for j=1:size(A,1)
           Dist(i,j)=N_distance(A(i,:),A(j,:));
       end
    end
    
    %  Weight Matrix
    W=ones(size(Dist));
    W(Dist>ee)=0; % Set too far away to
    W(Dist==0)=0;  % Avoid an edge to itself
    
    % Degree Matrix
    D=zeros(size(A,1));
    for i=1:size(A,1)
        D(i,i)=sum(W(i,:));
    end
    
    % Plot a graph from a weight matrix
    n=size(W,1);
    z=exp(((1:n)-1)*2i*pi/n);
    clf;
    plot(z,'o','MarkerFaceColor','k'); % Plot all nodes
    hold on;
    
    for i=1:size(W,1)
       for j=1:size(W,1);
           if (W(i,j)>0)
                plot([z(i),z(j)],'k');   % Plot edges
           end
       end
    end
    
    % Graph Laplacian
    L= D-W;
    [V,E]=eig(L);
end
