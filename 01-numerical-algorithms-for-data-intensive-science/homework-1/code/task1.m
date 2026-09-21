%% Problem b
% Define Matrix A
A = [1 2 2003 2005;
     2 2 2002 2004;
     3 2 2001 2003;
     4 7 7005 7012];

% % Initialize
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

%% Problem c
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

%% Problem d
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