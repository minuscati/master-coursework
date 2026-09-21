%% Problem 1
% Simulation of a Hidden Markov Model for Mobility Tracking
clear; clc; close all;

rng(2604);
% 1. Parameters Setup
dt = 0.5;       
alpha = 0.6;    
sigma = 0.5;    
m = 1000;      % Number of simulation steps

Phi_sub = [1, dt, dt^2/2; 
           0, 1, dt; 
           0, 0, alpha];

Psi_z_sub = [dt^2/2; dt; 0];
Psi_w_sub = [dt^2/2; dt; 1];

% Full matrices
Phi = blkdiag(Phi_sub, Phi_sub);
Psi_z = blkdiag(Psi_z_sub, Psi_z_sub);
Psi_w = blkdiag(Psi_w_sub, Psi_w_sub);

% 2. Driving Commands and Transitions
% The 5 possible states for Z_n
Z_set = [0, 3.5, 0, 0, -3.5;  
         0, 0, 3.5, -3.5, 0]; 

% Transition Probability Matrix P
P = (1/20) * [16, 1, 1, 1, 1;
              1, 16, 1, 1, 1;
              1, 1, 16, 1, 1;
              1, 1, 1, 16, 1;
              1, 1, 1, 1, 16];

% 3. Initialization
% Initial state X_0 (Normal distribution)
X0_mu = [0; 0; 0; 0; 0; 0];
X0_cov = diag([500, 5, 5, 200, 5, 5]);
X_n = mvnrnd(X0_mu, X0_cov)'; 

% Initial command Z_0 (Uniform distribution)
z_idx = randi([1, 5]);
Z_n = Z_set(:, z_idx);

% Store the trajectory (Position x1 is index 1, Position x2 is index 4)
traj = zeros(2, m+1);
traj(:, 1) = [X_n(1); X_n(4)];


% 4. Main Simulation Loop
for n = 1:m
    % Generate Gaussian noise W_{n+1}
    W_next = sigma * randn(2, 1);
    
    % Update motion state
    X_next = Phi * X_n + Psi_z * Z_n + Psi_w * W_next;
    
    % Update driving command: Select next Z based on matrix P
    % Use the current index to pick the next index
    z_idx = randsample(1:5, 1, true, P(z_idx, :));
    Z_next = Z_set(:, z_idx);
    
    % Store and update
    traj(:, n+1) = [X_next(1); X_next(4)];
    X_n = X_next;
    Z_n = Z_next;
end

% 5. Plotting
figure;
plot(traj(1,:), traj(2,:), 'LineWidth', 1.5);
hold on;
plot(traj(1,1), traj(2,1), 'ro', 'MarkerFaceColor', 'r'); % Start
plot(traj(1,end), traj(2,end), 'gs', 'MarkerFaceColor', 'g'); % End
grid on;
xlabel('Position X_1 (m)');
ylabel('Position X_2 (m)');
title('Simulated Target Trajectory');
legend('Path', 'Start', 'End');
axis equal;





%% Problem 3: SIS
load('Data\RSSI-measurements.mat'); 
load('Data\stations.mat');          

rng(26);
T_steps = size(Y, 2);                
N = 10000;                     
upsilon = 90; 
eta = 3; 
varsigma = 1.5;

% Initialization
X_p = mvnrnd(X0_mu, X0_cov, N)'; 
Z_idx = randi([1, 5], 1, N);     
log_W = log(ones(1, N) / N);

X_est_SIS = zeros(6, T_steps); 
ESS = zeros(1, T_steps);
cumP = cumsum(P, 2);             

disp('Running SIS... Watch for weight degeneracy.');
% Selected time points (n=0, 10, 15, 20, 50, 70, 100, 500)
sample_points = [1, 11, 16, 21, 51, 71, 101, 501]; 
figure('Name', 'Problem 3: Weight Histograms', 'Color', 'w');
plot_idx = 1;

for n = 1:T_steps
    % 1. Prediction
    if n > 1
        W_noise = sigma * randn(2, N);
        Z_vals = Z_set(:, Z_idx);
        X_p = Phi * X_p + Psi_z * Z_vals + Psi_w * W_noise;
    
        r = rand(1, N);
        for k = 1:5
            mask = (Z_idx == k);
            if any(mask)
                [~, next_states] = max(r(mask)' <= cumP(k, :), [], 2);
                Z_idx(mask) = next_states';
            end
        end
    end

    % 2. Weight Update
    log_L = zeros(1, N);
    pos_x = X_p([1, 4], :); 

    for l = 1:6
        dist = sqrt(sum((pos_x - pos_vec(:, l)).^2, 1));
        dist(dist < 0.1) = 0.1; 
        mu_RSSI = upsilon - 10 * eta * log10(dist);
        log_L = log_L - 0.5 * ((Y(l, n) - mu_RSSI) / varsigma).^2;
    end
    
    % Accumulate log weights
    log_W = log_W + log_L; 
    
    % Normalize
    W = exp(log_W - max(log_W));
    W = W / sum(W);
    log_W = log(W); 
    
    % 3. Compute efficient sample sizes and estimate status
    ESS(n) = 1 / sum(W.^2);
    X_est_SIS(:, n) = sum(X_p .* W, 2); 
    
    % 4. Plot histograms of the importance weights
    if ismember(n, sample_points)
        subplot(2, 4, plot_idx);
        histogram(W, 50, 'FaceColor', '#A2142F');
        title(['n = ', num2str(n-1), ', ESS = ', num2str(round(ESS(n), 2))]);
        xlabel('Normalized Weight'); ylabel('Count');
        plot_idx = plot_idx + 1;
    end
end


figure('Name', 'Problem 3: SIS Trajectory', 'Color', 'w');
plot(X_est_SIS(1,:), X_est_SIS(4,:), 'b-', 'LineWidth', 1.5); hold on;
plot(pos_vec(1,:), pos_vec(2,:), 'r^', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
grid on; axis equal;
title('SIS Estimated Trajectory and Station Locations');
xlabel('x_1 (m)'); ylabel('x_2 (m)');
legend('SIS Estimate', 'Base Stations');

saveas(gcf, 'problem 3.png');


%% Problem 4: SISR
load('Data\RSSI-measurements.mat');
load('Data\stations.mat');          

% set seed
rng(26);

T_steps = size(Y, 2);                
N = 10000;                     
upsilon = 90; eta = 3; 
% varsigma = 1.5;
varsigma = 2.2;

% Initialization
X_p = mvnrnd(X0_mu, X0_cov, N)'; 
Z_idx = randi([1, 5], 1, N);     
W = ones(1, N) / N; 

X_est_SISR = zeros(6, T_steps);
Z_map = zeros(1, T_steps);
RMSE = zeros(1, T_steps);
cumP = cumsum(P, 2);

disp('Running SISR for 501 steps...');

% n=1
log_L = zeros(1, N);
pos_x = X_p([1, 4], :); 
for l = 1:6
    dist = sqrt(sum((pos_x - pos_vec(:, l)).^2, 1));
    dist(dist < 0.1) = 0.1; 
    mu_RSSI = upsilon - 10 * eta * log10(dist);
    log_L = log_L - 0.5 * ((Y(l, 1) - mu_RSSI) / varsigma).^2;
end
W = exp(log_L - max(log_L));
W = W / sum(W);

% n>=2
for n = 2:T_steps
    % 1. Resampling
    ESS_current = 1 / sum(W.^2);
    if ESS_current < N/2
        edges = [0, cumsum(W)];
        edges(end) = 1; 
        u = (0:N-1)/N + rand()/N;
        idx = discretize(u, edges);
        
        X_p = X_p(:, idx);
        Z_idx = Z_idx(idx);
        W = ones(1, N) / N;
    end
        
    % 2. Prediction
    W_noise = sigma * randn(2, N);
    Z_vals = Z_set(:, Z_idx);
    X_p = Phi * X_p + Psi_z * Z_vals + Psi_w * W_noise;
    
    r = rand(1, N);
    for k = 1:5
        mask = (Z_idx == k);
        if any(mask)
            [~, next_states] = max(r(mask)' <= cumP(k, :), [], 2);
            Z_idx(mask) = next_states';
        end
    end
        
    % 3. Weight Update
    log_L = zeros(1, N);
    pos_x = X_p([1, 4], :); 
    for l = 1:6
        dist = sqrt(sum((pos_x - pos_vec(:, l)).^2, 1));
        dist(dist < 0.1) = 0.1; 
        mu_RSSI = upsilon - 10 * eta * log10(dist);
        log_L = log_L - 0.5 * ((Y(l, n) - mu_RSSI) / varsigma).^2;
    end
    
    log_W_unnorm = log(W) + log_L; 
    W = exp(log_W_unnorm - max(log_W_unnorm));
    W = W / sum(W);
    
    % Estimation 
    X_est_SISR(:, n) = sum(X_p .* W, 2); 
    
    Z_probs = zeros(1,5);
    for k = 1:5
        Z_probs(k) = sum(W(Z_idx == k));
    end
    [~, Z_map(n)] = max(Z_probs);
    
end
disp('Filtering Complete!');


% Plotting estimation
figure('Name', 'Problem 4: SISR Trajectory', 'Color', 'w');
plot(X_est_SISR(1, :), X_est_SISR(4, :), 'b-', 'LineWidth', 1.5); hold on;
plot(pos_vec(1, :), pos_vec(2, :), 'r^', 'MarkerFaceColor', 'r', 'MarkerSize', 8); 
grid on; axis equal;
title('Problem 4: SISR Estimated Trajectory and Station Locations');
xlabel('x_1 (m)'); ylabel('x_2 (m)');
legend('SIS Estimate', 'Base Stations');

% Plotting the Driving Command
figure('Name', 'Problem 4: Inferred Driver Commands', 'Color', 'w');
stairs(1:T_steps, Z_map, 'LineWidth', 1.5);
grid on;
xlim([0,510]);
ylim([0.5, 5.5]);
yticks(1:5);
yticklabels({'1: Hold', '2: East', '3: North', '4: West', '5: South'});
title('Inferred Most Probable Driving Command over Time');
xlabel('Time Step (n)');
ylabel('Driving Command (Z_n)');


%% Problem 5: SMC-based Model Calibration
close all;

% loading data
load('Data\RSSI-measurements-unknown-sigma.mat'); 
load('Data\stations.mat');          

T_steps = size(Y, 2);                
N = 10000;                     
upsilon = 90; eta = 3; sigma_w = 0.5;

% grid search
sigma_grid = 0.1:0.1:2.9;
ll_grid = zeros(1, length(sigma_grid));

disp('Running Model Calibration...');

% Main loop
for j = 1:length(sigma_grid)

    sig = sigma_grid(j);

    X_p = mvnrnd(X0_mu, X0_cov, N)'; 
    Z_idx = randi([1, 5], 1, N);     
    W = ones(1, N) / N; 

    ll_total = 0;

    % n=1
    log_L = zeros(1, N);
    pos_x = X_p([1,4], :); 

    for l = 1:6
        dist = sqrt(sum((pos_x - pos_vec(:,l)).^2, 1));
        dist(dist < 0.1) = 0.1;

        mu_RSSI = upsilon - 10 * eta * log10(dist);

        log_L = log_L ...
              - log(sqrt(2*pi)*sig) ...
              - 0.5*((Y(l,1) - mu_RSSI)/sig).^2;
    end

    W(W < 1e-300) = 1e-300;

    log_W_unnorm = log(W) + log_L;
    M = max(log_W_unnorm);

    ll_total = ll_total + (M + log(sum(exp(log_W_unnorm - M))));

    W = exp(log_W_unnorm - M);
    W = W / sum(W);

    % n>=2
    for n = 2:T_steps
      % 1. Resampling 
        ESS = 1 / sum(W.^2);
        if ESS < N/2
            W(isnan(W)) = 0;
            if sum(W) == 0, W = ones(1,N)/N; end
            W = W / sum(W);

            
            edges = [0, cumsum(W)];
            edges(end) = 1.000000000001; 
            u = rand() / N + (0:N-1) / N;
            [~, ~, idx] = histcounts(u, edges); 
            idx(idx == 0) = 1; 
            
            X_p = X_p(:, idx);
            Z_idx = Z_idx(idx);
            W = ones(1, N) / N;
        end

        % 2. Prediction 
        W_noise = sigma_w * randn(2, N);
        Z_vals = Z_set(:, Z_idx);
        X_p = Phi * X_p + Psi_z * Z_vals + Psi_w * W_noise;
        
        r = rand(1, N);
        current_cumP = cumP(Z_idx, :);
        Z_idx = sum(r' > current_cumP, 2)' + 1;

        % 3. Weight Update
        log_L = zeros(1, N);
        pos_x = X_p([1,4], :); 
        
        for l = 1:6
            dist = sqrt(sum((pos_x - pos_vec(:,l)).^2, 1));
            dist = max(dist, 0.1); 
            mu_RSSI = upsilon - 10 * eta * log10(dist);
            log_L = log_L - log(sig * 2.506628274631) ... % 2.5066 = sqrt(2*pi)
                    - 0.5 * ((Y(l,n) - mu_RSSI) / sig).^2;
        end
        
        W = max(W, 1e-300);
        log_W_unnorm = log(W) + log_L;
        M = max(log_W_unnorm);
        
        temp_sum = sum(exp(log_W_unnorm - M));
        ll_total = ll_total + (M + log(temp_sum));
        
        W = exp(log_W_unnorm - M) / temp_sum;
    end

    ll_grid(j) = ll_total / T_steps;

    fprintf('sigma = %.2f, LL = %.4f\n', sig, ll_grid(j));
end

% Optimal sigma
[max_ll, idx] = max(ll_grid);
best_sigma = sigma_grid(idx);

fprintf('============================\n');
fprintf('Best sigma = %.2f\n', best_sigma);

% Plot
figure;
plot(sigma_grid, ll_grid, 'b-o', 'LineWidth', 1.5);
hold on;
plot(best_sigma, max_ll, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
grid on;

xlabel('\varsigma');
ylabel('Normalized Log-Likelihood');
title('Log-Likelihood vs \varsigma');
legend('LL curve', 'MLE');


% Subsitituting best sigma
disp(['Running SMC with best_sigma = ', num2str(best_sigma)]);

% Initializing
sig = best_sigma;
X_p = mvnrnd(X0_mu, X0_cov, N)'; 
Z_idx = randi([1, 5], 1, N);     
W = ones(1, N) / N; 
ll_total = 0;

X_est = zeros(6, T_steps);

for n = 1:T_steps
    % 1. Resampling 
    ESS = 1 / sum(W.^2);
    if n > 1 && ESS < N/2
        edges = [0, cumsum(W)];
        edges(end) = 1.000001; 
        u = rand()/N + (0:N-1)/N;
        [~, ~, idx] = histcounts(u, edges);
        
        idx(idx == 0) = 1; 
        X_p = X_p(:, idx);
        Z_idx = Z_idx(idx);
        W = ones(1, N) / N;
    end

    % 2. Prediction 
    if n > 1
        W_noise = sigma_w * randn(2, N);
        Z_vals = Z_set(:, Z_idx);
        X_p = Phi * X_p + Psi_z * Z_vals + Psi_w * W_noise;
        
        
        r = rand(1, N);
        current_cumP = cumP(Z_idx, :); 
        Z_idx = sum(r' > current_cumP, 2)' + 1;
        Z_idx(Z_idx > 5) = 5;
    end

    % 3. Weight Update 
    log_L = zeros(1, N);
    pos_x = X_p([1,4], :); 
    
    for l = 1:6
        dist = sqrt(sum((pos_x - pos_vec(:,l)).^2, 1));
        dist = max(dist, 0.1); 
        mu_RSSI = upsilon - 10 * eta * log10(dist);
        
        log_L = log_L - log(sig * 2.50662827) - 0.5 * ((Y(l,n) - mu_RSSI) / sig).^2;
    end
   
    W_safe = max(W, 1e-300);
    log_W_unnorm = log(W_safe) + log_L;
    M = max(log_W_unnorm);
    
    sum_exp = sum(exp(log_W_unnorm - M));
    ll_total = ll_total + (M + log(sum_exp));
    
    W = exp(log_W_unnorm - M) / sum_exp;
    
    %  State Estimation
    X_est(:, n) = X_p * W';
end

final_ll = ll_total / T_steps;
fprintf('Final Evaluation with sigma %.2f: LL = %.4f\n', best_sigma, final_ll);

% plot
figure('Name', 'Problem 5: Trajectory under Best Sigma', 'Color', 'w');
plot(X_est(1, :), X_est(4, :), 'b-', 'LineWidth', 1.5); hold on;
plot(pos_vec(1, :), pos_vec(2, :), 'r^', 'MarkerFaceColor', 'r', 'MarkerSize', 8); 
grid on; axis equal;
xlabel('X Position (m)');
ylabel('Y Position (m)');
title(['Estimated Trajectory with \varsigma = ', num2str(best_sigma)]);
legend('Estimated Path', 'Stations');