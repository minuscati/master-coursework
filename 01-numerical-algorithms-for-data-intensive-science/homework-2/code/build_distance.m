function D=build_distance(items,w)
% items: a matrix where every column is an image of length 784
% w a weight vector (784*1) for the wegihted Euclidean product
n = size(items,2);
D = zeros(n, n); % Initialize the distance matrix
for i = 1:n
    for j = 1:n
        D(i, j) = sqrt(sum(w .* (items(:, i) - items(:, j)).^2)); % Compute weighted Euclidean distance
    end
end