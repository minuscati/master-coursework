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