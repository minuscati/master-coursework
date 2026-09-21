%% hw1_6 a
% read pictures
file_sam = imread("roundabout_snapshots/roundabout_snapshots_0001.png");
imfloat=double(file_sam);
[n1, n2, n3]=size(imfloat);
n=n1*n2*n3; 

A = zeros(n, 56);

for k = 1:56
    filename = sprintf("roundabout_snapshots/roundabout_snapshots_%04d.png", k);
    frame_k = imread(filename);
    % resize every pic to same size
    frame_k = imresize(frame_k, [n1 n2]);

    imfloat=double(frame_k);
    v=reshape(imfloat,[],1);
    A(:,k)=v;
end



%% a
p = 20;
[Q, R, error] = Algorithm1_greedy_truncated(A,p);

figure;
semilogy(1:p,error(1:p),'-o','LineWidth',1.5);
% xlabel; ylabel; title

%% b
Q10 = Q(:,1:10);
R10 = Q10'*A;
A_approx10 = Q10*R10;
Q20 = Q(:,1:20);
% R20 = R(1:20,:);
R20 = Q20'*A;
A_approx20 = Q20*R20;
E10 = A- A_approx10;
E20 = A- A_approx20;

ERR10 = vecnorm(E10);
ERR20 = vecnorm(E20);


figure;
plot(1:k, ERR10, '-o', 'LineWidth', 1.5)
hold on
plot(1:k, ERR20, '-o', 'LineWidth', 1.5)
xlabel('snapshot index')
ylabel('Error per snapshot')
title('Comparison of approximation errors per image for rank 10 and 20')
legend('p=10','p=20')
gird on



%% c
[U_hat,S,V] = svd(Q'*A,"econ"); %!! R又不能直接用 用Q'A
U = Q*U_hat;
K = [1;2;5];
AF = zeros(n,length(K));

for k_idx = 1:length(K)
    App_fi = U(:,1:K(k_idx)) * S(1:K(k_idx), 1:K(k_idx)) * V(:, 1:K(k_idx))';
    AF(:,k_idx) = App_fi(:,1);
end

% column to image
figure;
image_r1 = uint8(reshape(AF(:,1),n1,n2,n3));
imshow(image_r1);

figure;
image_r2 = uint8(reshape(AF(:,2),n1,n2,n3));
imshow(image_r2);

figure;
image_r5 = uint8(reshape(AF(:,3),n1,n2,n3));
imshow(image_r5);

%%d
% rank=1;
App_1 = U(:,1:1) * S(1:1, 1:1) * V(:, 1:1)';
makevideo(App_1, n1, n2, "video_r1")

% rank=5;
App_5 = U(:,1:5) * S(1:5, 1:5) * V(:, 1:5)';
makevideo(App_5, n1, n2, "video_r5")

% rank=20;
App_20 = U(:,1:20) * S(1:20, 1:20) * V(:, 1:20)';
makevideo(App_20, n1, n2, "video_r20")

