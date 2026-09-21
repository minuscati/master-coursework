%% Question 1
%%%% TASK B) %%%%
% Define Matrix A
A = [1 2 2003 2005;
     2 2 2002 2004;
     3 2 2001 2003;
     4 7 7005 7012];

% Initialize
[m,n] = size(A);
p = min(m,n);
[Q, R, error] = Algorithm1_gs(A);

fprintf('Error ||Aj|| after iterations: \n');
disp(error);
fprintf('Error ||Aj|| after 3 iterations: \n');
disp(error(3));

%Plot
figure;
semilogy(1:p, error, '-o', 'LineWidth', 2, 'MarkerSize', 6);
grid on;
title('Convergence of Residual Norm ||A_j||');
xlabel('j');
ylabel('||A_j||');
xlim([1, p]);
saveas(gcf, 'images\q1_b.png');


%%%% TASK C) %%%%
Z = load_mat_hw1(1000,100);  % n=5, p=4
% Initialize
[m,n] = size(Z);
p = min(m,n);
[Q1, R1, error] = Algorithm1_gs(Z);

%Plot
figure;
semilogy(1:p, error, '-o', 'LineWidth', 2, 'MarkerSize', 6);
grid on;
title('Convergence of Residual Norm ||Z_j||');
xlabel('j');
ylabel('||Z_j||');
xlim([1, p]);
saveas(gcf, 'images\q1_c.png');


%%%% TASK D) %%%%
% greedy version of alg1
Z = load_mat_hw1(1000,100);  % Load data

% Initialize
[m,n] = size(Z);
p = min(m,n);
[Q, R, error] = Algorithm1_gs_greedy(Z);

% Plot
figure;
semilogy(1:p, error, '-o', 'LineWidth', 2, 'MarkerSize', 6);
grid on;
title('Convergence of Residual Norm ||Z_j|| (Greedy version)');
xlabel('j');
ylabel('||Z_j||');
xlim([1, p]);
saveas(gcf, 'images\q1_d.png');

%% Question 3
%%%% TASK A) %%%%
% Generate matrix A
A = load_mat_hw1(1000,100);
d = svd(A);
% 2norm
k_2norm = find(d < 1e-10, 1) -1;

% plot error
figure;
semilogy(d, 'o-', 'LineWidth', 1.5);
grid on;
hold on;

yline(1e-10, 'r--', 'Threshold: 10^{-10}', 'LabelVerticalAlignment', 'bottom');

xlabel('k');
ylabel('Singular Value');
title('Singular Value Decay Curve');

% find the lowest rank
k_min = find(d < 1e-10, 1) - 1;
fprintf('The lowest rank is %d\n within the threshold', k_min);


%%%% TASK B) %%%%
[Q, R, error] = Algorithm1_gs(A);
[Uhat, S, V] = svd(R, 'econ');
U = Q * Uhat;

for k=1:100
    A_approx = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';
    err = norm(A - A_approx);
    if err < 1e-10
        fprintf('Approximation error for rank %d: %e\n', k, err);
        break
    end
end
% test
approxError = norm(A - A_approx);
rank(A_approx)


%% Question 5
% load data
file_sam = imread("testbild_snapshots/testbild_snapshots_0001.png");
imfloat=double(file_sam);
n1=size(imfloat,1); n2=size(imfloat,2); n3=3;
n=n1*n2*n3; 
V = zeros(n, 27);
for k = 1:27
    filename = sprintf("testbild_snapshots/testbild_snapshots_%04d.png", k);
    frame_k = imread(filename);
    imfloat=double(frame_k);
    v=reshape(imfloat,n,1);
    V(:,k)=v;
end

%%%% TASK C) %%%%
A = V;
u = A(:,1);
v_T = ones(1,27);
E = A - u * v_T;
tol2 = norm(E); %2-norm
tol = norm(E,"fro"); %frobrenius norm
fprintf('The norm ||A - uv^T|| is: %f\n', tol);

%%%% TASK D) %%%%
% qr+svd algorithm
[Q, R, error] = Algorithm1_gs(A); 
[Uhat, S, V_svd] = svd(R, 'econ'); 
U = Q * Uhat;

% initialize
s_v = diag(S);
n_rank = 0;

for k = 1:size(S, 2)
    if k < size(S, 2)
        err = s_v(k+1); 
    else
        err = 0;
    end
    
    if err <= tol2
        n_rank = k;
        fprintf('With tol = %e, the numerical rank is: %d\n', tol2, k);
        break;
    end
end

% test the result
A_approx = U(:, 1:n_rank) * S(1:n_rank, 1:n_rank) * V_svd(:, 1:n_rank)';
actual_err = norm(A - A_approx);
fprintf('Verified 2-norm error: %e\n', actual_err);


%%%% TASK E) %%%%
% Time Comparison
% SVD in matlab
tic;
[U_mat, S_mat, V_mat] = svd(A, 0); 
t_matlab = toc;

% SVD via algorithm 1
tic;
[Q, R, err] = Algorithm1_gs(A); 
[Uhat, S_reduced, V_reduced] = svd(R, 'econ');
U_mine = Q * Uhat;
t_mine = toc;

fprintf('MATLAB svd(A,0) time: %.6f seconds\n', t_matlab);
fprintf('Our SVD via alg1 time: %.6f seconds\n', t_mine);

%-----
% Accuracy Comparison
% extract the optimal SVD from Matalb's svd
A_mat_1 = U_mat(:,1) * S_mat(1,1) * V_mat(:,1)';

% Calculate the error
err_c = norm(A - u * v_T, 'fro');       % simple rank-one approximation in (c)
err_d = norm(A - A_approx, 'fro');     % svd via Algo 1 in (d) 
err_e = norm(A - A_mat_1, 'fro');    % Matalb's svd

fprintf('Error (c) [First frame]: %e\n', err_c);
fprintf('Error (d) [First frame]: %e\n', err_d);
fprintf('Error (e) [Optimal SVD]: %e\n', err_e);

%% Question 6
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

%% Question 7
%%%% TASK A) %%%%%
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

% [Q, R, error] = Algorithm1_gs_greedy(A);

% initialize
rng(100,'twister');
k = 5; %rank = 5
s_vec = 0:25; %number of oversample
errors = zeros(length(s_vec), 1);
rel_error = zeros(length(s_vec), 1);
[m, n] = size(A);

for i = 1:length(s_vec)
    s = s_vec(i);
    A_approx = random_svd(A, k, s) ;
    errors(i) = norm(A - A_approx,'fro'); 
    rel_error(i) = norm(A - A_approx, 'fro') / norm(A, 'fro');
end

% Plot
figure;
semilogy(s_vec, errors, '-o', 'LineWidth', 1.5);
xlabel('Oversampling parameter s');
ylabel('Approximation Error');
title('Error vs Oversampling Parameter s');
grid on;


%%%% TASK B) %%%%%
n_runs = 30;
k = 5;
s = 15;
[m, n] = size(A);

% initialize
times = zeros(n_runs, 3);
Errors = zeros(n_runs, 3);
errs = zeros(3, 1);

% comparing
for r = 1:n_runs
    % --- (i) Randomized SVD ---------------------------------------
    tic;
    Approx1 = random_svd(A, k, s);
    % fprintf('Matrix Approx1 has rank %d \n', rank(Approx1)); %test approximation
    times(r, 1) = toc;
    Errors(r,1) = norm(A - Approx1, 'fro');
    
    
    % --- (ii) Greedy Version ---------------------------------------
    tic;
    [Q, ~, ~] = Algorithm1_greedy_truncated(A, k); 
    Q_k = Q(:, 1:k);
    R_k = Q_k' * A; 
    [Uhat, S, V_svd] = svd(R_k, 'econ');
    Approx2 = (Q_k * Uhat) * S * V_svd';
    % fprintf('Matrix Approx2 has rank %d \n', rank(Approx2));
    times(r, 2) = toc;
    Errors(r,2) = norm(A - Approx2, 'fro');
    
    
    % --- (iii) Matlab svd(A, 0) -------------------------------------
    tic;
    [U3, S3, V3] = svd(A, 0);
    Approx3 = U3(:, 1:k) * S3(1:k, 1:k) * V3(:, 1:k)';
    % fprintf('Matrix Approx3 has rank %d \n', rank(Approx3));
    times(r, 3) = toc;
    Errors(r,3) = norm(A - Approx3, 'fro');
end

avg_times = mean(times);
avg_error = mean(Errors);

% print output
fprintf('Method\t\tAvg Time (s)\tError\n');
fprintf('random_SVD\t%.6f\t%.4e\n', avg_times(1), avg_error(1));
fprintf('Greedy\t\t%.6f\t%.4e\n', avg_times(2), avg_error(2));
fprintf('Matlab SVD\t%.6f\t%.4e\n', avg_times(3), avg_error(3));

%% Question 8
% cd('~/Documents/Education/Master/NumAlgo/HW1');

file_items = 'zalando_items.mat';
load(file_items);
A = item5;
clear file_items item0 item1 item2 item3 item4 item5 item6 item7 item8 item9;

%%%% TASK A) %%%%
k = rank(A); % = 769

% SVD:
[U S V] = svd(A, "econ"); % best rank approximation
RelErr_svd = zeros(k, 1);
for i=1:k
    Asvd_i = U(:, 1:i)*S(1:i, 1:i)*V(:, 1:i)';
    RelErr_svd(i) = norm(A-Asvd_i, 'fro') / norm(A, 'fro');
end
% at what rank is the 25% accuracy of approximation reached?
svd_accuracy = find(RelErr_svd <= 0.25, 1, 'first'); % = 126

% ID:
[C,Z]=ID_col(A,k);
RelErr_id = zeros(k, 1);
for i=1:k
    Aid_i = C(:, 1:i)*Z(1:i, :);
    RelErr_id(i) = norm(A-Aid_i, 'fro') / norm(A, 'fro');
end
id_accuracy = find(RelErr_id <= 0.25, 1, 'first'); % = 768

% semilogy plot
x = 1:k;
y1 = RelErr_svd;
y2 = RelErr_id;
figure
semilogy(x, y1, '-o', 'LineWidth', 1.5) % svd's relative error
hold on
semilogy(x, y2, '-s', 'LineWidth', 1.5) % id's relative error
xlabel('Rank k')
ylabel('Relative Error')
legend('SVD error', 'ID error', 'Location', 'best')
grid on


%%%% TASK B) %%%%
% plotting first three singular vectors
zalando_plot(U(:, 1));
zalando_plot(U(:, 2));
zalando_plot(U(:, 3));

% plotting first three ID vectors
zalando_plot(C(:, 1));
zalando_plot(C(:, 2));
zalando_plot(C(:, 3));


%% Functions used
function [Q, R, err] = Algorithm1_gs(A)
    [m,n] = size(A);
    p = min(m,n);

    Q = zeros(m, p);
    R = zeros(p, n);
    err = zeros(p,1);

    A_ccur =A;

    for j = 1:p
       v = A_ccur(:,j);
       normv = norm(v);
       q_j = v / normv;

       r_j_T =  q_j.' * A_ccur;

       Q(:,j) = q_j;
       R(j,:) = r_j_T;

       A_next = A_ccur - q_j * r_j_T;
       err(j) = norm(A_next, "fro")
       A_ccur = A_next; % Update A_ccur for the next iteration
    end
end

function [Q, R, error] = Algorithm1_gs_greedy(A)
    % Initialize
    [m,n] = size(A);
    p = min(m,n);
    Q = zeros(m, p);
    R = zeros(p, n);
    error = zeros(p,1);
    A_ccur = A;
    
    % Loop for Greedy Alg 1
    for j = 1:p
       % Calculate norms of remaining columns (j to n)
       current_norms = vecnorm(A_ccur(:, j:n), 2, 1);
       %finding corresponding index of max_norm
       [~, max_idx] = max(current_norms);
       pivot_idx = j + max_idx - 1;
       
       % Swap columns in Z_ccur
       if pivot_idx ~= j
           A_ccur(:, [j, pivot_idx]) = A_ccur(:, [pivot_idx, j]);
       end
      
       v = A_ccur(:,j); %max column
       normv = norm(v);
       q_j = v / normv;
       r_j_T =  q_j.' * A_ccur; 
       Q(:,j) = q_j;
       R(j,:) = r_j_T;
       
       A_next = A_ccur - q_j * r_j_T;
       
       error(j) = norm(A_next, 'fro');
       A_ccur = A_next; % Update for the next iteration
    end
end


function [Q, R, error] = Algorithm1_greedy_truncated(A, k)
    % Initialize
    [m,n] = size(A);
    p = min(m,n);
    Q = zeros(m, p);
    R = zeros(p, n);
    error = zeros(p,1);
    A_ccur = A;
    
    % Loop for Greedy Alg 1
    for j = 1:k
       current_norms = vecnorm(A_ccur(:, j:n), 2, 1);
       [~, max_idx] = max(current_norms);
       pivot_idx = j + max_idx - 1;
       
       % Swap columns in Z_ccur
       if pivot_idx ~= j
           A_ccur(:, [j, pivot_idx]) = A_ccur(:, [pivot_idx, j]);
       end
      
       v = A_ccur(:,j); %max column
       normv = norm(v);
       q_j = v / normv;
       r_j_T =  q_j.' * A_ccur; 
       Q(:,j) = q_j;
       R(j,:) = r_j_T;
       
       A_next = A_ccur - q_j * r_j_T;
       
       error(j) = norm(A_next, 'fro');
       A_ccur = A_next; % Update for the next iteration
    end
end


% randomized svd
function A_approx = random_svd(A, k, s)
    % k:rank; s:oversample number
    [m, n] = size(A);
    p = k + s; 
     % Stage A
    Omega = randn(n, p);
    Y = A * Omega;   
  
    [Q, ~] = qr(Y, 0);
    
    % Stage B
    B = Q' * A;
    [U_hat, S, V] = svd(B, 'econ');
    U = Q * U_hat;
    
    % Approxiamtion
    A_approx = U(:, 1:k) * S(1:k, 1:k) * V(:, 1:k)';
end

