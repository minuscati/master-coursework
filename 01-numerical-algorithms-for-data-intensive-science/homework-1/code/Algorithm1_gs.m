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

