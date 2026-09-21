%% Problem a
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


%% Problem b
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