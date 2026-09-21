%% Task 2 %%%%
n0=5;p=3;
randn('seed',0);
% rng(0,'twister');
c=2;ee=2.5;
A1=randn(n0,p)+c*[1,0,0];
A2=randn(n0,p)+c*[0,1,0];
A3=randn(n0,p)+c*[0,0,1];
A4=[0,0,0];
A=[A1;A2;A3;A4];
%% b)
[~,~,E1]=Simi_graph(A,c,ee);
%% c)
c=3;
ee=2.5;
A1=randn(n0,p)+c*[1,0,0];
A2=randn(n0,p)+c*[0,1,0];
A3=randn(n0,p)+c*[0,0,1];
A4=[0,0,0];
A=[A1;A2;A3;A4];
[~,~,E2]=Simi_graph(A,c,ee);
%% d)
c=3.1;
ee=2.8;
A1=randn(n0,p)+c*[1,0,0];
A2=randn(n0,p)+c*[0,1,0];
A3=randn(n0,p)+c*[0,0,1];
A4=[0,0,0];
A=[A1;A2;A3;A4];
[L3,V3,E3]=Simi_graph(A,c,ee);
% the second eigenvector:
EV2=V3(:,2);
% test
% L3*EV2
%plot
figure
plot(1:16,EV2,'o',LineWidth=1.5);
xlabel('i');
ylabel('v2(i)');
%% e)
EV3=V3(:,3);
plot(EV2,EV3,'o');
xlabel('v2(i)');
ylabel('v3(i)');
%%

%%%% TASK 3 %%%%
%% Given code:
load('bengali_cleanup.mat');
A=imread('bengali_map.png');
jv=[102,280,10];
% figure of the three measument stations
figure; clf;
imshow(A); hold on;
plot(y_coords(jv),x_coords(jv),'r*') % Note: x and y reversed since
                                     % images have swapped x and y axis.
% fig 2                                     
figure; clf; 
plot(tv,timeseries(jv(1),:),'-'); hold on
plot(tv,timeseries(jv(2),:),'--')
plot(tv,timeseries(jv(3),:),'-.')
%% a) 
% figure of all the stations on the map
figure; clf;
imshow(A); hold on;
plot(y_coords, x_coords,'r.')
%% b) distance matrix
[n_r, n_c] = size(timeseries); % 937x100
dist=zeros(n_r, n_r); % 937x937
for i = 1:n_r
    for j = 1:n_r
        dist(i,j) = norm(timeseries(i, :) - timeseries(j, :), 2);
    end
end
% distances of measurment stations 102, 280 and 10 
% dist(jv, jv) 
%% c) unweighted adjacency matrix
% uncomment the option you need to use for TASK C or TASK E
NN = 3; % num. of Nearest Nbh (TASK C)
% NN = 2; % num. of NN (TASK E)
W = zeros(n_r, n_r); % unweighted adjacency matrix
for i = 1:n_r
    [~, idx] = sort(dist(i,:), 'ascend');
    neighbors = idx(2:NN+1); % because always idx(1)=0
    W(i, neighbors) = 1; % 1 if connection exists
end
% final adjacency matrix
W = max(W, W'); % OR version: 1 if i near j and j near i
% graph
G = graph(W);
figure; clf;
plot(G);
title("OR version of kNN graph")
%% d) Spectral clustering algorithm
% Degree matrix
D = diag(sum(W, 2)); % Degree matrix
k = 7; % num of clusters
L = D - W; % graph Laplacian matrix
[X, lambda] = eigs(L, k, 'smallestreal');
Cl = kmeans(X, k);
% 7 maps with clusters
for c = 1:k
    figure; clf;
    imshow(A); hold on;
    % plot only stations in cluster c
    cluster_points = (Cl == c);
    plot(y_coords(cluster_points), ...
         x_coords(cluster_points), ...
         'r.', 'MarkerSize', 12);
    title(['Cluster ', num2str(c)]);
end
%%

%%%% TASK 4 %%%%
A = load('zalando_clustering.mat').items;
C = load('zalando_clustering.mat').correct;
load('zalando_clustering.mat');
%% a)
randn('seed',0);
rand_indices1 = randperm(1000, 9);

figure;
for i = 1:9
    subplot(3, 3, i);
    zalando_plot(items(:, rand_indices1(i)));
    title(['Index: ', num2str(rand_indices1(i))]);
end


rand_indices2 = randperm(1000, 9);
figure;
for i = 1:9
    subplot(3, 3, i);
    zalando_plot(items(:, rand_indices2(i)));
    title(['Index: ', num2str(rand_indices2(i))]);
end
%% b)
w = ones(784,1);
D = build_distance(A,w);
% first group of comparison
col1 = items(:,1);
% d=D(2:1000,1);
[v1,i1] = min(D(2:1000,1));
col2 = items(:,i1+1);
% second group of comparison
col3 = items(:,2);
% d=D(2:1000,1);
d1 = D(1,2);
[v2,i2] = min(D(3:end,2));
if d1<v2
    col4 = items(:, 1);
else
    col4 = items(:,i2+2);
end
% plot
figure;
subplot(2,2,1);
zalando_plot(col1);
xlabel('(i) Picture of the first item');
subplot(2,2,2);
zalando_plot(col2);
xlabel('(ii) Image closest to the first item');
subplot(2,2,3);
zalando_plot(col3);
xlabel('(iii) Picture of the second item');
subplot(2,2,4);
zalando_plot(col4);
xlabel('(iv) Image closest to the second item');
%% c)
% weight matrix
n = size(D, 1);
W = zeros(n, n);
alpha=0.5;
for i=1:n
sgma=std(D(:,i));
for j=1:n
W(i,j)=exp(-alpha*D(i,j)^2/sgma^2);
end
end
W=W-diag(diag(W));
W=(W+W')/2;
% Degree matrix
Deg =zeros(size(D,1));
for i=1:size(D,1)
    Deg(i,i)=sum(W(i,:));
end
% Calculate Laplacian
L = Deg - W;
% first two eigvectors
[V, ~] = eigs(L, 2, 1e-15);
% The second eigenvector
fiedler_vector = V(:, 2);
% Clustering based on the sign
% idx = (fiedler_vector > 0)+1;
figure;
plot(fiedler_vector/norm(fiedler_vector),'*');
ylabel('v2/norm(v2)');
ylim([-0.01,0.01]);
me = median(fiedler_vector);
idx2 = (fiedler_vector<me)+1;
per = sum((idx2==C))/1000;
if per>0.5
    per = per;
else
    per = 1-per;
end
co = (idx2==C); %incorrect shown as 0
II = find(idx2~=C); % index of incorrect ones
hw2_plot_bad_skeleton(items,II,correct,idx2,D);
%% -----e
% x=2; %12 wrong - 98.8%
x=20;  %6 wrong - 99.4%
W_weight=ones(28,28); W_weight(12:16,:)=x;
w_weight=W_weight(:);
w_weight=w_weight/norm(w_weight);
zalando_plot(w_weight);
title("x=20, W\_weight=ones(28,28)")
D_weight = build_distance(A,w_weight);
% weight matrix
n = size(D_weight, 1);
W = zeros(n, n);
alpha=0.5;
for i=1:n
sgma=std(D_weight(:,i));
for j=1:n
W(i,j)=exp(-alpha*D_weight(i,j)^2/sgma^2);
end
end
W=W-diag(diag(W));
W=(W+W')/2;
% Degree matrix
Deg =zeros(size(D_weight,1));
for i=1:size(D_weight,1)
    Deg(i,i)=sum(W(i,:));
end
% Calculate Laplacian
L = Deg - W;
% first two eigvectors
[V, ~] = eigs(L, 2, 1e-15);
% The second eigenvector
fiedler_vector = V(:, 2);
% Clustering based on the sign
% idx = (fiedler_vector > 0)+1;
plot(fiedler_vector/norm(fiedler_vector),'*');
ylabel('v2/norm(v2)');
me = median(fiedler_vector);
idx2 = (fiedler_vector<me)+1;
per = sum((idx2==C))/1000;
if per>0.5
    per = per;
else
    idx2 = (fiedler_vector>me)+1;
    per = sum((idx2==C))/1000;
end
co = (idx2==C); %incorrect shown as 0
II = find(idx2~=C); % index of incorrect ones
hw2_plot_bad_skeleton(items,II,correct,idx2,D);
