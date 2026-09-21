function y = fftx_eveneqodd(x)
n = length(x);
f_even = x(1:2:n-1);
f_odd = x(2:2:n);

if isequal(f_even, f_odd)
    omega = exp(-2i*pi/n);
    k = (0:n/2-1)';
    w = omega.^k;
    u = fftx(f_even);
    v = w.*u;

    y = [u + v; u - v];
else
    y = fftx(x);
end