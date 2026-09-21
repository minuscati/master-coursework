% test a)
X = [1;1;2;2];
fft_X = fftx(X);

%% b)
b = tan(2*(1:16)');
fft_b = fftx(b);

%% c)
% building Fn matrix

K=20;
p=13;
nv = 2.^(1:p);
rng(25);

T_n = zeros(length(nv),K);
T_f = zeros(length(nv),K);

for k=1:K
    for idx=1:length(nv)
        n = nv(idx);
        b = rand(n, 1);
        % construncting Fn
        omega = exp(-2i * pi / n);
        F = zeros(n, n);
        vec = 0:n-1;
        F = omega.^(vec' * vec);
    
        % Naive
        tic;
        y1 = F*b;
        T_n(idx,k) = toc;
        
        % Fft
        tic;
        y2 = fftx(b);
        T_f(idx,k) = toc;
    end
end

tn=mean(T_n,2); 
tf=mean(T_f,2);


figure;
plot(nv, tn, '--r', 'LineWidth', 1.5); hold on;
plot(nv, tf, '-b', 'LineWidth', 1.5);
grid on;
xlabel('n');
ylabel('CPU Time');
title('Performance Comparison of Naive vs FFT');
legend('Naive', 'FFT');

%% d)
%test
X = [1;1;2;2];
tic
fft_X = fftx_eveneqodd(X);
t1=toc;

tic
fft_X = fftx(X);
t2=toc;

% 
K=20;
p=13;
nv = 2.^(1:p);
rng(25);

T_n = zeros(length(nv),K);
T_f = zeros(length(nv),K);
T_feo = zeros(length(nv),K);

for k=1:K
    for idx=1:length(nv)
        n = nv(idx);
        b = rand(n/2, 1);
        b = reshape([b, b]', [n, 1]);
        
        % construncting Fn
        omega = exp(-2i * pi / n);
        F = zeros(n, n);
        vec = 0:n-1;
        F = omega.^(vec' * vec);
    
        % Naive
        tic;
        y1 = F*b;
        T_n(idx,k) = toc;
        
        % Fft
        tic;
        y2 = fftx(b);
        T_f(idx,k) = toc;

        % Fft_eve=odd
        tic;
        y3 = fftx_eveneqodd(b);
        T_feo(idx,k) = toc;
    end
end

tn=mean(T_n,2); 
tf=mean(T_f,2);
tfeo=mean(T_feo,2);


figure;
plot(nv, tn, '--r', 'LineWidth', 1.5); hold on;
plot(nv, tf, '-b', 'LineWidth', 1.5); hold on;
plot(nv, tfeo, '-g', 'LineWidth', 1.5);
grid on;
xlabel('n');
ylabel('CPU Time');
title('Performance Comparison');
legend('Naive', 'FFT', 'FFT\_optimized');
