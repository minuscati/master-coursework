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




