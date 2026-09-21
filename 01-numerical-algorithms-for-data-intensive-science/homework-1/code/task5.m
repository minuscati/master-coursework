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

%% Problem c
A = V;
u = A(:,1);
v_T = ones(1,27);
E = A - u * v_T;
tol2 = norm(E); %2-norm
tol = norm(E,"fro"); %frobrenius norm
fprintf('The norm ||A - uv^T|| is: %f\n', tol);

%% Problem d
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


%% Problem e
% 1. SVD in matlab
tic;
[U_mat, S_mat, V_mat] = svd(A, 0); 
t_matlab = toc;

% 2. SVD via algorithm 1
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













