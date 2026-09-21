%% Task2_4
A = load('zalando_clustering.mat').items;
C = load('zalando_clustering.mat').correct;
load('zalando_clustering.mat');

% a
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

% b
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


%% c
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
ylim([-0.003,0.003]);


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


%% ---------e
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
