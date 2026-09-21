%% HA2 - Problem 1(c)
clear; clc;

% load data
tau_table = readtable('Data\coal-mine.csv'); 
tau = table2array(tau_table); 
tau = tau(:)'; 

N = 20000;          
vartheta = 1;       
rho = 0.05;         

% d = 2, 3, 4, 5
num_breakpoints_list = [1, 2, 3, 4];

figure('Position', [100, 100, 1200, 800]); 

for idx = 1:length(num_breakpoints_list)
    num_bp = num_breakpoints_list(idx);
    d = num_bp + 1; 

    % Initialization
    t_curr = zeros(1, d+1);
    t_curr(1) = 1851; t_curr(d+1) = 1963;
    avg_dis = (t_curr(d+1) - t_curr(1)) / d;
    for j = 2:d
        t_curr(j) = t_curr(j-1) + avg_dis;
    end
    theta_curr = 1;
    lambda_curr = ones(1, d);

    t_history = zeros(N, d+1);

    % MCMC main loop
    for i = 1:N
        % 1. Gibbs update theta
        theta_curr = gamrnd(2*d + 2, 1 / (vartheta + sum(lambda_curr)));

        % 2. Gibbs update lambda
        n = zeros(1, d);
        for j = 1:d
            n(j) = sum(tau >= t_curr(j) & tau < t_curr(j+1));
        end
        for j = 1:d
            lambda_curr(j) = gamrnd(n(j) + 2, 1 / (t_curr(j+1) - t_curr(j) + theta_curr));
        end

        % 3. MH update t
        for j = 2:d
            R = rho * (t_curr(j+1) - t_curr(j-1));
            t_prop = t_curr(j) + (2 * rand() - 1) * R;

            if t_prop > t_curr(j-1) && t_prop < t_curr(j+1)
                n_j1_curr = sum(tau >= t_curr(j-1) & tau < t_curr(j));
                n_j_curr  = sum(tau >= t_curr(j)   & tau < t_curr(j+1));
                n_j1_prop = sum(tau >= t_curr(j-1) & tau < t_prop);
                n_j_prop  = sum(tau >= t_prop      & tau < t_curr(j+1));

                log_post_curr = n_j1_curr * log(lambda_curr(j-1)) - lambda_curr(j-1) * (t_curr(j) - t_curr(j-1)) + log(t_curr(j) - t_curr(j-1)) ...
                              + n_j_curr  * log(lambda_curr(j))   - lambda_curr(j)   * (t_curr(j+1) - t_curr(j))   + log(t_curr(j+1) - t_curr(j));
                log_post_prop = n_j1_prop * log(lambda_curr(j-1)) - lambda_curr(j-1) * (t_prop - t_curr(j-1)) + log(t_prop - t_curr(j-1)) ...
                              + n_j_prop  * log(lambda_curr(j))   - lambda_curr(j)   * (t_curr(j+1) - t_prop)   + log(t_curr(j+1) - t_prop);

                if log(rand()) < (log_post_prop - log_post_curr)
                    t_curr(j) = t_prop;
                end
            end
        end
        t_history(i, :) = t_curr;
    end

    % Plot
    subplot(4, 1, idx);
    hold on;
    for bp_idx = 2:d
        plot(t_history(:, bp_idx), 'DisplayName', ['t\_', num2str(bp_idx)]);
    end
    hold off;
    title(['Trace plot:', num2str(num_bp), ' breakpoints']);
    xlabel('Iteration');
    ylabel('Year');
    grid on;
    % if num_bp > 1; legend('Location', 'best'); end
end


%% Problem 1(d)
clear; clc;

% load data
tau_table = readtable('Data\coal-mine.csv'); 
tau = table2array(tau_table); 
tau = tau(:)'; 

N = 20000;          
rho = 0.05;
num_bp=2; %d=3
d = num_bp + 1;

vartheta_list = [0.1,1,10,100];
figure('Position', [100, 100, 1200, 800]); 

for var = 1:length(vartheta_list)
    vartheta = vartheta_list(var);

    % Initialization
    t_curr = zeros(1, d+1);
    t_curr(1) = 1851; t_curr(d+1) = 1963;
    avg_dis = (t_curr(d+1) - t_curr(1)) / d;
    for j = 2:d
        t_curr(j) = t_curr(j-1) + avg_dis;
    end
    theta_curr = 1;
    lambda_curr = ones(1, d);

    t_history = zeros(N, d+1);

    % MCMC main loop
    for i = 1:N
        % 1. Gibbs update theta
        theta_curr = gamrnd(2*d + 2, 1 / (vartheta + sum(lambda_curr)));

        % 2. Gibbs update lambda
        n = zeros(1, d);
        for j = 1:d
            n(j) = sum(tau >= t_curr(j) & tau < t_curr(j+1));
        end
        for j = 1:d
            lambda_curr(j) = gamrnd(n(j) + 2, 1 / (t_curr(j+1) - t_curr(j) + theta_curr));
        end

        % 3. MH update t
        for j = 2:d
            R = rho * (t_curr(j+1) - t_curr(j-1));
            t_prop = t_curr(j) + (2 * rand() - 1) * R;

            if t_prop > t_curr(j-1) && t_prop < t_curr(j+1)
                n_j1_curr = sum(tau >= t_curr(j-1) & tau < t_curr(j));
                n_j_curr  = sum(tau >= t_curr(j)   & tau < t_curr(j+1));
                n_j1_prop = sum(tau >= t_curr(j-1) & tau < t_prop);
                n_j_prop  = sum(tau >= t_prop      & tau < t_curr(j+1));

                log_post_curr = n_j1_curr * log(lambda_curr(j-1)) - lambda_curr(j-1) * (t_curr(j) - t_curr(j-1)) + log(t_curr(j) - t_curr(j-1)) ...
                              + n_j_curr  * log(lambda_curr(j))   - lambda_curr(j)   * (t_curr(j+1) - t_curr(j))   + log(t_curr(j+1) - t_curr(j));
                log_post_prop = n_j1_prop * log(lambda_curr(j-1)) - lambda_curr(j-1) * (t_prop - t_curr(j-1)) + log(t_prop - t_curr(j-1)) ...
                              + n_j_prop  * log(lambda_curr(j))   - lambda_curr(j)   * (t_curr(j+1) - t_prop)   + log(t_curr(j+1) - t_prop);

                if log(rand()) < (log_post_prop - log_post_curr)
                    t_curr(j) = t_prop;
                end
            end
        end
        t_history(i, :) = t_curr;
    end

    % Plot
    subplot(4, 1, var);
    hold on;
    for bp_idx = 2:d
        plot(t_history(:, bp_idx), 'DisplayName', ['t\_', num2str(bp_idx)]);
    end
    title(['Trace plot:','\vartheta=',num2str(vartheta)]);
    xlabel('Iteration');
    ylabel('Year');
    grid on;
    % if num_bp > 1; legend('Location', 'best'); end
end

%% Problem 1 (e)
clear; clc;

load data
tau_table = readtable('Data\coal-mine.csv'); 
tau = table2array(tau_table); 
tau = tau(:)'; 

N = 20000;          
num_bp=2; %d=3
d = num_bp + 1;
vartheta = 1;

rho_list = [0.005,0.05,0.5,5];
figure('Position', [100, 100, 1200, 800]); 

for r = 1:length(rho_list)
    rho = rho_list(r);

    Initialization
    t_curr = zeros(1, d+1);
    t_curr(1) = 1851; t_curr(d+1) = 1963;
    avg_dis = (t_curr(d+1) - t_curr(1)) / d;
    for j = 2:d
        t_curr(j) = t_curr(j-1) + avg_dis;
    end
    theta_curr = 1;
    lambda_curr = ones(1, d);

    t_history = zeros(N, d+1);

    MCMC main loop
    for i = 1:N
        1. Gibbs update theta
        theta_curr = gamrnd(2*d + 2, 1 / (vartheta + sum(lambda_curr)));

        2. Gibbs update lambda
        n = zeros(1, d);
        for j = 1:d
            n(j) = sum(tau >= t_curr(j) & tau < t_curr(j+1));
        end
        for j = 1:d
            lambda_curr(j) = gamrnd(n(j) + 2, 1 / (t_curr(j+1) - t_curr(j) + theta_curr));
        end

        3. MH update t
        for j = 2:d
            R = rho * (t_curr(j+1) - t_curr(j-1));
            t_prop = t_curr(j) + (2 * rand() - 1) * R;

            if t_prop > t_curr(j-1) && t_prop < t_curr(j+1)
                n_j1_curr = sum(tau >= t_curr(j-1) & tau < t_curr(j));
                n_j_curr  = sum(tau >= t_curr(j)   & tau < t_curr(j+1));
                n_j1_prop = sum(tau >= t_curr(j-1) & tau < t_prop);
                n_j_prop  = sum(tau >= t_prop      & tau < t_curr(j+1));

                log_post_curr = n_j1_curr * log(lambda_curr(j-1)) - lambda_curr(j-1) * (t_curr(j) - t_curr(j-1)) + log(t_curr(j) - t_curr(j-1)) ...
                              + n_j_curr  * log(lambda_curr(j))   - lambda_curr(j)   * (t_curr(j+1) - t_curr(j))   + log(t_curr(j+1) - t_curr(j));
                log_post_prop = n_j1_prop * log(lambda_curr(j-1)) - lambda_curr(j-1) * (t_prop - t_curr(j-1)) + log(t_prop - t_curr(j-1)) ...
                              + n_j_prop  * log(lambda_curr(j))   - lambda_curr(j)   * (t_curr(j+1) - t_prop)   + log(t_curr(j+1) - t_prop);

                if log(rand()) < (log_post_prop - log_post_curr)
                    t_curr(j) = t_prop;
                end
            end
        end
        t_history(i, :) = t_curr;
    end

    Plot
    subplot(4, 1, r);
    hold on;
    for bp_idx = 2:d
        plot(t_history(:, bp_idx), 'DisplayName', ['t\_', num2str(bp_idx)]);
    end
    title(['Trace plot:','\rho=',num2str(rho)]);
    xlabel('Iteration');
    ylabel('Year');
    grid on;
    if num_bp > 1; legend('Location', 'best'); end
end

%% Problem 2: HMC vs Random Walk Metropolis-Hastings
clear; clc; close all;
set(0, 'defaultfigurecolor', [1 1 1]);

%% 1. Data loading and parameter setup
% Read observations y
y = csvread('hmc-observations.csv');

% Known hyperparameters
sigma = 2;
Sigma = [5, 0; 0, 0.5];
Sigma_inv = inv(Sigma);
n = length(y);

%% 2. Function Handles for Potential U(theta) and its Gradient grad_U(theta)
% U(theta) = -ln f(theta|y) (ignoring constants)
U = @(theta) 0.5 * (theta' * Sigma_inv * theta) + ...
    (1/(2*sigma^2)) * sum((y - (theta(1)^2 + theta(2)^2)).^2);

% grad_U(theta)
grad_U = @(theta) (Sigma_inv * theta) - ...
    (2/sigma^2) * sum(y - (theta(1)^2 + theta(2)^2)) * theta;

%% 3. HMC and RWMH Setup
N_samples = 10000;
theta_init = [0; 1.5]; % Initial state

% HMC Tuning Parameters
epsilon = 0.05; % Leapfrog step size
L = 20;         % Number of leapfrog steps

% RWMH Tuning Parameter
zeta = 0.15;    % Standard deviation for Gaussian random walk

% Storage for chains
samples_HMC = zeros(2, N_samples);
samples_RWMH = zeros(2, N_samples);
samples_HMC(:,1) = theta_init;
samples_RWMH(:,1) = theta_init;

% Acceptance counters
acc_HMC = 0;
acc_RWMH = 0;

%% 4. Run HHMC
for i = 2:N_samples
    theta_curr = samples_HMC(:, i-1);

    % Sample momentum
    v_curr = randn(2, 1);

    % Record initial Hamiltonian
    H_curr = U(theta_curr) + 0.5 * (v_curr' * v_curr);

    % Leapfrog integration
    theta_prop = theta_curr;
    v_prop = v_curr;

    % Half-step momentum
    v_prop = v_prop - (epsilon/2) * grad_U(theta_prop);
    % Alternating full-steps for position and momentum
    for l = 1:(L-1)
        theta_prop = theta_prop + epsilon * v_prop;
        v_prop = v_prop - epsilon * grad_U(theta_prop);
    end
    % Final full-step position and half-step momentum
    theta_prop = theta_prop + epsilon * v_prop;
    v_prop = v_prop - (epsilon/2) * grad_U(theta_prop);

    % Hamiltonian of proposed state
    H_prop = U(theta_prop) + 0.5 * (v_prop' * v_prop);

    % Metropolis Accept/Reject
    alpha = exp(H_curr - H_prop);
    if rand() < alpha
        samples_HMC(:, i) = theta_prop;
        acc_HMC = acc_HMC + 1;
    else
        samples_HMC(:, i) = theta_curr;
    end
end

%% 5. Run RWMH
for i = 2:N_samples
    theta_curr = samples_RWMH(:, i-1);

    % Random walk proposal
    theta_prop = theta_curr + zeta * randn(2, 1);

    % Calculate posterior ratio
    U_curr = U(theta_curr);
    U_prop = U(theta_prop);

    alpha = exp(U_curr - U_prop);
    if rand() < alpha
        samples_RWMH(:, i) = theta_prop;
        acc_RWMH = acc_RWMH + 1;
    else
        samples_RWMH(:, i) = theta_curr;
    end
end

%% 6. Results and Performance Comparison
acc_rate_HMC = acc_HMC / N_samples;
acc_rate_RWMH = acc_RWMH / N_samples;
fprintf('HMC Acceptance Rate: %.2f%%\n', acc_rate_HMC * 100);
fprintf('RWMH Acceptance Rate: %.2f%%\n', acc_rate_RWMH * 100);

% Plot 2D histograms
figure('Position', [100, 100, 1200, 500], 'Name', 'Posterior Density Estimation');

subplot(1, 2, 1);
histogram2(samples_HMC(1,:), samples_HMC(2,:), 50, 'DisplayStyle', 'tile');
colorbar; colormap('parula');
title(sprintf('HMC Posterior (Acc Rate: %.2f)', acc_rate_HMC));
xlabel('\theta_1'); ylabel('\theta_2'); axis equal; xlim([-2.5 2.5]); ylim([-2.5 2.5]);

subplot(1, 2, 2);
histogram2(samples_RWMH(1,:), samples_RWMH(2,:), 50, 'DisplayStyle', 'tile');
colorbar; colormap('parula');
title(sprintf('RWMH Posterior (Acc Rate: %.2f)', acc_rate_RWMH));
xlabel('\theta_1'); ylabel('\theta_2'); axis equal; xlim([-2.5 2.5]); ylim([-2.5 2.5]);

% Plot Autocorrelation Function (ACF) to show mixing efficiency
figure('Position', [150, 150, 1200, 400], 'Name', 'Autocorrelation (Mixing efficiency)');
max_lag = 100;

subplot(1, 2, 1);
[acf_HMC, lags] = autocorr(samples_HMC(1,:), 'NumLags', max_lag);
stem(lags, acf_HMC, 'Marker', 'none');
title('HMC \theta_1 Autocorrelation');
xlabel('Lag'); ylabel('ACF'); ylim([-0.2 1]);

subplot(1, 2, 2);
[acf_RWMH, ~] = autocorr(samples_RWMH(1,:), 'NumLags', max_lag);
stem(lags, acf_RWMH, 'Marker', 'none');
title('RWMH \theta_1 Autocorrelation');
xlabel('Lag'); ylabel('ACF'); ylim([-0.2 1]);


%% 7. Parameter Tuning Study (HMC vs RWMH)
% HMC Grid Search (eps vs L)
N_tune = 2000; 

eps_vals = [0.01, 0.05, 0.1, 0.2, 0.3];
L_vals = [2, 5, 10, 20, 40];
acc_HMC_grid = zeros(length(eps_vals), length(L_vals));

for i = 1:length(eps_vals)
    for j = 1:length(L_vals)
        eps_i = eps_vals(i);
        L_j = L_vals(j);

        theta_tmp = [0; 1.5]; 
        acc_tmp = 0;
        rng(2026); 

        for t = 1:N_tune
            theta_curr = theta_tmp;
            p_curr = randn(2, 1);
            H_old = U(theta_curr) + 0.5 * (p_curr' * p_curr);

            % leapfrog integrator
            theta_prop = theta_curr;
            p_prop = p_curr - 0.5 * eps_i * grad_U(theta_prop);
            for step = 1:(L_j-1)
                theta_prop = theta_prop + eps_i * p_prop;
                p_prop = p_prop - eps_i * grad_U(theta_prop);
            end
            theta_prop = theta_prop + eps_i * p_prop;
            p_prop = p_prop - 0.5 * eps_i * grad_U(theta_prop);

            H_new = U(theta_prop) + 0.5 * (p_prop' * p_prop);

            % Metropolis Accept/Reject
            if log(rand()) < (H_old - H_new)
                theta_tmp = theta_prop;
                acc_tmp = acc_tmp + 1;
            end
        end
        acc_HMC_grid(i,j) = acc_tmp / N_tune;
        fprintf('HMC (eps=%.3f, L=%2d) - Acceptance Rate: %.2f%%\n', eps_i, L_j, acc_HMC_grid(i,j)*100);
    end
end

% RWMH Tuning (zeta)
zeta_vals = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0];
acc_RWMH_list = zeros(length(zeta_vals), 1);

for i = 1:length(zeta_vals)
    zeta_i = zeta_vals(i);
    theta_tmp = [0; 1.5];
    acc_tmp = 0;
    rng(42);

    for t = 1:N_tune
        theta_curr = theta_tmp;
        theta_prop = theta_curr + zeta_i * randn(2, 1);

        U_old = U(theta_curr);
        U_new = U(theta_prop);

        if log(rand()) < (U_old - U_new)
            theta_tmp = theta_prop;
            acc_tmp = acc_tmp + 1;
        end
    end
    acc_RWMH_list(i) = acc_tmp / N_tune;
    fprintf('RWMH (zeta=%.2f) - Acceptance Rate: %.2f%%\n', zeta_i, acc_RWMH_list(i)*100);
end

%% Visualization
figure('Position', [100, 100, 1000, 450], 'Name', 'Parameter Tuning Results');

% HMC Heatmap
subplot(1, 2, 1);
heatmap(string(L_vals), string(eps_vals), acc_HMC_grid, 'Colormap', parula, 'CellLabelFormat', '%.2f');
title('HMC Acceptance Rate');
xlabel('Number of leapfrog steps (L)');
ylabel('Step size (\epsilon)');

% RWMH Line Graph
subplot(1, 2, 2);
plot(zeta_vals, acc_RWMH_list, '-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
title('RWMH Acceptance Rate');
xlabel('Proposal Std (\zeta)');
ylabel('Acceptance Rate');
ylim([0 1]);

