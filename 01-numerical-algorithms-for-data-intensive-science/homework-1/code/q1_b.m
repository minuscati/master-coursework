% Define Matrix A
A = [1, 2, 2003, 2005;
     2, 2, 2002, 2004;
     3, 2, 2001, 2003;
     4, 7, 7005, 7012];

[m, n] = size(A);
p = min(m, n);

% 初始化变量 / Initialize variables
A_curr = A;       % A_0
errors = zeros(p, 1); % 用于存储每一步的误差 / Store norms ||A_j||
Q = zeros(m, p);

% 2. 循环算法 Algorithm 1 / Loop Algorithm 1
for j = 1:p
    % --- Step 1: Compute q_j (Normalize) ---
    % 取出当前矩阵的第 j 列 / Take the j-th column
    v = A_curr(:, j);
    
    % 计算范数 / Compute norm
    norm_v = norm(v);
    
    % 处理数值零的情况（秩亏）/ Handle rank deficiency (numerical zero)
    if norm_v < 1e-10
        q_j = zeros(m, 1);
    else
        q_j = v / norm_v;
    end
    
    % --- Step 2: Compute r_j^T (Projection) ---
    % 计算投影系数 / Projection coefficients
    % 注意：MATLAB中 ' 表示共轭转置，.' 表示非共轭转置（实数矩阵通用）
    r_j_T = q_j.' * A_curr; 
    
    % --- Step 3: Update A (Elimination) ---
    % A_j = A_{j-1} - q_j * r_j^T
    % 从剩余矩阵中减去投影分量 / Subtract projection from current matrix
    A_next = A_curr - (q_j * r_j_T);
    
    % 记录残差矩阵的 Frobenius 范数 / Record Frobenius norm of residual
    errors(j) = norm(A_next, 'fro');
    
    % 更新变量 / Update for next iteration
    A_curr = A_next;
    Q(:, j) = q_j;
end

% 3. 打印结果 / Print Results
fprintf('Error ||A_j|| after each iteration:\n');
disp(errors);

fprintf('Error after i=3 is: %e\n', errors(3));

% 4. 绘图 / Plotting
figure;
semilogy(1:p, errors, '-o', 'LineWidth', 2, 'MarkerSize', 6);
grid on;
title('Convergence of Residual Norm ||A_j||');
xlabel('Iteration (j)');
ylabel('||A_j|| (Log Scale)');
xlim([1, p]);