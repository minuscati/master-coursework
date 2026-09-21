function Similarity_graph(A, ee)
    % A: 输入数据矩阵 (n x d)
    % ee: 距离阈值 (epsilon)
    
    n = size(A, 1);
    Dist = zeros(n);
    
    %% 1. 计算距离矩阵 (Distance Matrix)
    for i = 1:n
       for j = 1:n
           % 确保 N_distance 函数已定义
           Dist(i,j) = N_distance(A(i,:), A(j,:));
       end
    end
    
    %% 2. 构建权重矩阵/邻接矩阵 (Weight Matrix)
    W = double(Dist <= ee); % 距离小于等于 ee 的设为 1
    W = W - eye(n);        % 将对角线置为 0，避免自环
    W(W < 0) = 0;          % 确保没有负值
    
    %% 3. 绘图部分 (二选一)
    
    figure; % 开启新窗口，避免覆盖之前的图
        %% Plot a graph from a weight matrix
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


    % plot(graph(W));
    % % --- 方案 A: 使用 Matlab 内置图论库 (推荐，布局更智能) ---
    % G = graph(W);
    % p = plot(G);
    % p.MarkerSize = 7;
    % p.MarkerColor = 'r';
    % p.EdgeColor = [0.5 0.5 0.5]; % 灰色线条
    % title(['Similarity Graph (\epsilon = ', num2str(ee), ')']);
    
    %{ 
    % --- 方案 B: 手动圆形布局 (如果你需要特定展示节点排列) ---
    clf;
    z = exp(((1:n)-1) * 2i * pi / n); % 将节点分布在复平面的单位圆上
    plot(real(z), imag(z), 'ko', 'MarkerFaceColor', 'r'); 
    hold on;
    for i = 1:n
       for j = i+1:n % 只遍历上三角，避免重复连线
           if W(i,j) > 0
                plot([real(z(i)), real(z(j))], [imag(z(i)), imag(z(j))], 'k-');
           end
       end
    end
    axis equal; axis off;
    %}
end

% %% Distance Marix
% function Similarity_graph(A,ee)
%     Dist=zeros(size(A,1));
%     for i=1:size(A,1)
%        for j=1:size(A,1)
%            Dist(i,j)=N_distance(A(i,:),A(j,:));
%        end
%     end
% 
%     %%  Weight Matrix
%     W=ones(size(Dist));
%     W(Dist>ee)=0; % Set too far away to
%     W(Dist==0)=0;  % Avoid an edge to itself
% 
%     %% Degree Matrix
%     D=zeros(size(A,1));
%     for i=1:size(A,1)
%         D(i,i)=sum(W(i,:));
%     end
% 
%     %% Plot a graph from a weight matrix
%     n=size(W,1);
%     z=exp(((1:n)-1)*2i*pi/n);
%     clf;
%     plot(z,'o','MarkerFaceColor','k'); % Plot all nodes
%     hold on;
% 
%     for i=1:size(W,1)
%        for j=1:size(W,1);
%            if (W(i,j)>0)
%                 plot([z(i),z(j)],'k');   % Plot edges
%            end
%        end
%     end
% 
%     %% Matlab graph plotting function
%     clf; plot(graph(W));
% end