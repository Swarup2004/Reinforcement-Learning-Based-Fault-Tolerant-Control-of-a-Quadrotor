% plot_results.m
% Evaluates trained agent under healthy and fault conditions

cd('C:\Users\ONGC\RL_for_quadrotors')
addpath('01_params')
addpath('03_rl_env')
addpath('04_agent')

% Load trained agent
load('05_train\trained_agent.mat', 'agent', 'trainingStats')
fprintf('Trained agent loaded\n')

% Define test scenarios
scenarios = {
    'Healthy (no fault)',         [1.0;1.0;1.0;1.0];
    'Motor1: 50% LoE',            [0.5;1.0;1.0;1.0];
    'Motor1: 80% LoE (severe)',   [0.2;1.0;1.0;1.0];
    'Motor2: 50% LoE',            [1.0;0.5;1.0;1.0];
};

colors = {'b','r','m','g'};
N = 1000;
results = cell(size(scenarios,1),1);

% Run each scenario
for s = 1:size(scenarios,1)
    env = QuadrotorEnv();
    env.Lambda = scenarios{s,2};

    % Fixed initial condition for fair comparison
    env.State = [0.5; 0; 0; 0; 0; 0; 0; 0];

    z_hist   = zeros(N,1);
    phi_hist = zeros(N,1);
    tht_hist = zeros(N,1);
    r_hist   = zeros(N,1);
    a_hist   = zeros(N,4);

    obs = [env.State(1)-env.z_target; env.State(2:8); env.State(1)];
    for k = 1:N
        action = getAction(agent,{obs});
        a = action{1};
        [obs, reward, done, ~] = step(env, a);
        z_hist(k)     = obs(9);
        phi_hist(k)   = obs(3);
        tht_hist(k)   = obs(5);
        r_hist(k)     = reward;
        a_hist(k,:)   = a';
        if done && k < N
            z_hist(k+1:end)   = z_hist(k);
            phi_hist(k+1:end) = phi_hist(k);
            tht_hist(k+1:end) = tht_hist(k);
            break
        end
    end
    results{s} = struct('z',z_hist,'phi',phi_hist,...
        'theta',tht_hist,'r',r_hist,'a',a_hist);
    fprintf('Scenario: %s | Total reward: %.1f | Final z: %.3f m\n',...
        scenarios{s,1}, sum(r_hist), z_hist(end))
end

t = (1:N)' * 0.01;

% Plot
figure('Name','Fault Tolerant Control Results','Position',[100 100 1200 800])

subplot(3,1,1)
hold on
yline(1.0,'k--','Target z=1m','LineWidth',1.5)
for s=1:size(scenarios,1)
    plot(t, results{s}.z, colors{s}, 'LineWidth',1.5,...
        'DisplayName', scenarios{s,1})
end
xlabel('Time [s]'); ylabel('Altitude z [m]')
title('Altitude Tracking under Healthy and Fault Conditions')
legend('show','Location','southeast'); grid on

subplot(3,1,2)
hold on
yline(0,'k--'); 
for s=1:size(scenarios,1)
    plot(t, rad2deg(results{s}.phi), colors{s},'LineWidth',1.5,...
        'DisplayName', scenarios{s,1})
end
xlabel('Time [s]'); ylabel('Roll angle [deg]')
title('Roll Angle (RL compensates asymmetric thrust from fault)')
legend('show','Location','northeast'); grid on

subplot(3,1,3)
hold on
yline(0,'k--');
for s=1:size(scenarios,1)
    plot(t, rad2deg(results{s}.theta), colors{s},'LineWidth',1.5,...
        'DisplayName', scenarios{s,1})
end
xlabel('Time [s]'); ylabel('Pitch angle [deg]')
title('Pitch Angle')
legend('show','Location','northeast'); grid on

% Print summary table
fprintf('\n=== PERFORMANCE SUMMARY ===\n')
fprintf('%-30s  %10s  %10s  %10s\n','Scenario','TotalRew','FinalZ','MaxRoll')
fprintf('%s\n', repmat('-',1,65))
for s=1:size(scenarios,1)
    fprintf('%-30s  %10.1f  %10.3f  %10.2f\n',...
        scenarios{s,1}, sum(results{s}.r),...
        results{s}.z(end), rad2deg(max(abs(results{s}.phi))))
end
