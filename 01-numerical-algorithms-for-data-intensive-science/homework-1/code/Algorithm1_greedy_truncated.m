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