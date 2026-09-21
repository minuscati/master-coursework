%% hw1_7 a
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

%%---------------
%%---------------

%% b
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

% print
fprintf('Method\t\tAvg Time (s)\tError\n');
fprintf('random_SVD\t%.6f\t%.4e\n', avg_times(1), avg_error(1));
fprintf('Greedy\t\t%.6f\t%.4e\n', avg_times(2), avg_error(2));
fprintf('Matlab SVD\t%.6f\t%.4e\n', avg_times(3), avg_error(3));