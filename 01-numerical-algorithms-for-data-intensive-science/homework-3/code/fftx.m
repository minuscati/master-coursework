function y = fftx(x);
% FFT computes the fast Fourier transform of x(1:n)
x = x(:);
n = length(x);
omega = exp(-2i*pi/n);
if rem(n,2) == 0
    %Recursive divide and conquer
    k = (0:n/2-1)';
    w = omega.^k;
    u = fftx(x(1:2:n-1));
    v = w.*fftx(x(2:2:n));
    y = [u + v; u - v];
else
    % Generate the Fourier Matrix
    j = 0:n-1;
    k = j';
    F = omega.^(k*j);
    y =F*x;
end