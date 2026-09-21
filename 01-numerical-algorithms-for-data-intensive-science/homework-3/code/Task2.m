cd("/home/elena/Documents/Education/Master/NumAlgo/HW3");

% loading the sound
[X, fs] = audioread("hw3_terrible_sound_with_hidden_message.wav");
% sample of length 105472
% playing the sound
% sound(X, fs);

% a)
Y = fft(X); % applying fast fourier trasnfor
% Y is an array of complex numbers representing the phase and magnitude 
% of every frequency in the sound

T = 1/fs;   % Sampling period       
L = length(X);  % Length of signal
t = (0:L-1)*T;  % Time vector

% semilogy plot of magnitude
figure
semilogy(fs/L*(0:L-1),abs(Y));
title("Complex Magnitude of FFT Spectrum")
xlabel("f (Hz)")
ylabel("|fft(X)|")

% b)
% finding noise frequencies
thr = 0.01 * max(abs(Y)); % 1% of the maximum value
Ynew = Y;
Ynew(abs(Y) >= thr) = 0;

% semilogy plot without noise
figure
semilogy(fs/L*(0:L-1),abs(Ynew));
title("Complex Magnitude of FFT Spectrum After Removing Noise")
xlabel("f (Hz)")
ylabel("|fft(X)|")

Xnew = X;
Xnew = ifft(Ynew); % applying inverse fast fourier trasnform
sound(Xnew, fs);
% the hidden message is: "Numerics is fun"

% c)
% naive Discrete Fourier Transform implementation
samples = [1000, 2000, 4000]; % vector of sample sizes
times = zeros(size(samples)); % vector of recorded time
for i = 1:length(samples)
    n = samples(i);
    f = randn(n, 1); % dummy random signal for ith sample    
    tic;    
    % Naive DFT calculation: O(N^2) matrix multiplication
    l = 0:n-1;
    j = 0:n-1;
    Z = exp(-1i * 2 * pi / n .* (l' * j)); % Z matrix
    hatf = Z * f; % hatf = Z_n*f
    times(i) = toc;
end
times
% the sample of size 1000 took 0.0258 seconds
% the sample of size 2000 took 0.1220 seconds
% the sample of size 4000 took 0.6102 seconds

% problems' size with 30min computational time
% computational cost: T = c * N^2
c = times(3) / (samples(3)^2); 
target = 1800; 
N = sqrt(target / c); % 217,250 
T_dft = c * length(X)^2; % 424.2372 seconds for DFT
% T_dft/60; % 7.0706 minutes

% since the problem addressed in the task has a signal size of 105,472, we
% could have performed the problem within 30 minutes. Specifically, 
% we would need approximately 7 minutes. However, it is very unconvenient,
% because we can perfomr the same procedure with a cheaper computational
% cost: O(n*log(n)).




























