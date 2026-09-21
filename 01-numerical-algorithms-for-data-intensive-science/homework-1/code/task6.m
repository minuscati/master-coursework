% cd(["~/Documents/Education/Master/NumAlgo/HW1/"]);

% Loading images into a matrix V
file_name = fullfile('roundabout_snapshots', '*.png'); % created search pattern
list_images = dir(file_name); % list of obj. 
n_cols = length(list_images); % number of columns = number of snapshots

for k = 1 : n_cols
    V_uint8 = imread(fullfile('roundabout_snapshots', list_images(k).name));
    V_float(:, k) = double(V_uint8(:)); % switch to standard floating points
end

% info for display image in task c
info = imfinfo(fullfile('roundabout_snapshots', list_images(1).name));
h = info.Height;
w = info.Width;

V = V_float;

% clearing space
clear V_float V_uint8;

%%%% TASK A) %%%%

% parameters
[m, n] = size(V);
p = 20;

% step 0:
A = V; % A_0 = V
Q = zeros(m, p);
R = zeros(p, n);

%  to store approximation errors per iterations
Aj_norm = zeros(p,1);

for j = 1:p
    % start of greedy Algorithm 1
    % step 1:
    % computation ||w_k|| for every column k of A
    norms_w = vecnorm(A);

    % find index i with max norm
    [max_norm, i] = max(norms_w);

    % step 2:
    % compute q_j from max norm i
    % q_j= w_i / ||w_i||
    q = A(:, i) / max_norm;
    
    % step 3: 
    % compute r_j' = q_j * A_{j-1}
    r = q'*A;
    
    % store into matrices
    Q(:, j) = q;
    R(j, :) = r;

    % step 4:
    % update approximation error A for the next iteration
    A = A - q * r; 
    % end of greedy Algorithm 1

    % norm of appr. A_j (residual error per-iteration)
    Aj_norm(j) = norm(A, 'fro');

end

figure
plot(1:p, Aj_norm, '-o')
xlabel('Number of iterations')
ylabel('||A_j||_F')
grid on

%%%% TASK B) %%%%

% rank p = 20
V_QR = Q * R;
error_snapshot = vecnorm(V - V_QR);

% rank t = 10
t = 10;                
[m, n] = size(V);

Q2 = zeros(m, t);
R2 = zeros(t, n);
A2 = V;

for j = 1:t
    norms_w = vecnorm(A2);
    [~, i] = max(norms_w);
    q = A2(:, i) / norm(A2(:,i));
    r = q' * A2;
    Q2(:, j) = q;
    R2(j, :) = r;
    A2 = A2 - q * r;
end

% per-image errors
V_QR2 = Q2 * R2;
error_snapshot2 = vecnorm(V - V_QR2); 

% plot of the approximation error per snapshot
figure
plot(1:n, error_snapshot2, '-o', 'LineWidth',1.5)
hold on
plot(1:n, error_snapshot, '-s', 'LineWidth',1.5)
xlabel('Snapshots index')
ylabel('Error per snapshot')
title('Comparison of approximation errors per image for different ranks')
legend('p = 10','p = 20')
grid on

% clearing space
clear A A2 Q Q2 R R2;

%%%% TASK C) %%%%

% computing SVD of V_appr = U*S*W'
[U, S, W] = svd(V_QR, 0);

% define the rank
k = 1;
% k = 2;
% k = 5;

% select the top k components directly (No loop needed)
U_k = U(:, 1:k);       % First k columns of U
S_k = S(1:k, 1:k);     % Top-left kxk block of Sigma
W_k = W(:, 1:k);       % First k columns of W

% compute the approximation
V_svd = U_k * S_k * W_k';
clear U_k S_k W_k;

% display the reconstructed image from the SVD approximation
img_k = reshape(V_svd(:, 1), [h, w, 3]);
figure;
imshow(rescale(img_k)) % rescale normalizes RGB SVD per channel
title(['Reconstruction Rank k = ', num2str(k)]);

%%%% TASK D %%%%%
makevideo(V_svd, h, w, 'video1.avi')
