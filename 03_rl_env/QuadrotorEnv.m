classdef QuadrotorEnv < rl.env.MATLABEnvironment
%  QuadrotorEnv  —  MATLAB RL Environment for Fault-Tolerant Quadrotor
%
%  STATE (8x1):  [z, vz, phi, p, theta, q, psi, r]
%    z      = altitude [m]
%    vz     = vertical velocity [m/s]
%    phi    = roll  angle [rad]
%    p      = roll  rate  [rad/s]
%    theta  = pitch angle [rad]
%    q      = pitch rate  [rad/s]
%    psi    = yaw   angle [rad]
%    r      = yaw   rate  [rad/s]
%
%  OBSERVATION (9x1): state + altitude_error
%    [z_err, vz, phi, p, theta, q, psi, r, z]
%    z_err = z - z_target  (what the agent needs to correct)
%
%  ACTION (4x1): normalised motor commands in [-1, 1]
%    Actual omega_i^2 = omega_h^2 + action_i * delta_scale
%    At action=0 all motors run at hover speed => T = mg
%
%  REWARD:
%    r = -w1*(z_err^2) - w2*(vz^2) - w3*(phi^2+theta^2) - w4*(sum(a^2))
%    + hover_bonus if |z_err|<0.1
%    - crash_penalty if |phi|>60deg or |theta|>60deg or z<-1
%
%  FAULT MODEL:
%    50% of episodes: one random motor gets lambda in [0.3,1.0]
%    50% of episodes: all motors healthy (lambda=[1,1,1,1])

    properties
        % Physical parameters (loaded from quad_params.m)
        m        = 1.2722
        g        = 9.81
        kT       = 2.980e-6
        kQ       = 1.140e-7
        l_arm    = 0.225
        Ixx      = 0.0121
        Iyy      = 0.0121
        Izz      = 0.0221
        omega_h  = 0        % hover rotor speed [rad/s] set in constructor
        dt       = 0.01     % timestep [s]

        % Target altitude
        z_target = 1.0      % [m]

        % Reward weights
        w_z      = 2.0      % altitude error weight
        w_vz     = 0.5      % vertical velocity weight
        w_angle  = 1.0      % attitude error weight
        w_action = 0.05     % action effort weight

        % Episode limits
        MaxSteps     = 1000 % 10 sec at dt=0.01
        StepCount    = 0
        MaxTilt      = 1.0472  % 60 degrees [rad]
        MinAltitude  = -1.0    % crash if below this [m]

        % Action scaling
        DeltaScale   = 0    % set in constructor: 50% of omega_h^2

        % Fault state
        Lambda       = ones(4,1)  % [motor1;motor2;motor3;motor4]

        % Current state [z;vz;phi;p;theta;q;psi;r]
        State        = zeros(8,1)
    end

    methods
        %--- CONSTRUCTOR ---
        function this = QuadrotorEnv()
            % Observation: 9 values
            % [z_err, vz, phi, p, theta, q, psi, r, z]
            ObsInfo = rlNumericSpec([9 1], ...
                'LowerLimit', [-5;-10;-pi;-15;-pi;-15;-pi;-15;-2], ...
                'UpperLimit', [ 5; 10; pi; 15; pi; 15; pi; 15; 8]);
            ObsInfo.Name = 'QuadrotorObs';
            ObsInfo.Description = '[z_err,vz,phi,p,theta,q,psi,r,z]';

            % Action: 4 motor commands in [-1,1]
            ActInfo = rlNumericSpec([4 1], ...
                'LowerLimit', -ones(4,1), ...
                'UpperLimit',  ones(4,1));
            ActInfo.Name = 'MotorCmds';
            ActInfo.Description = 'Normalised delta from hover per motor';

            this = this@rl.env.MATLABEnvironment(ObsInfo, ActInfo);

            % Compute hover speed
            this.omega_h   = sqrt(this.m * this.g / (4 * this.kT));
            this.DeltaScale = this.omega_h^2 * 0.5;

            % Initial state: hovering at target
            this.State    = zeros(8,1);
            this.State(1) = this.z_target;
        end

        %--- STEP ---
        function [Obs, Reward, IsDone, Info] = step(this, Action)
            Info = [];

            % Clamp action to [-1,1]
            a = double(Action);
            a = max(-1, min(1, a));

            % Decode: omega_i^2 = omega_h^2 + a_i * DeltaScale
            omega2 = this.omega_h^2 + a * this.DeltaScale;
            omega2 = max(0, omega2);

            % Apply fault: F_eff_i = lambda_i * omega_i^2
            F = this.Lambda .* omega2;

            % Thrust/torque mixing
            % T         = kT*(F1+F2+F3+F4)
            % tau_phi   = kT*l*(-F1+F2+F3-F4)
            % tau_theta = kT*l*(-F1-F2+F3+F4)
            % tau_psi   = kQ*(-F1+F2-F3+F4)
            T         = this.kT * (F(1)+F(2)+F(3)+F(4));
            tau_phi   = this.kT*this.l_arm*(-F(1)+F(2)+F(3)-F(4));
            tau_theta = this.kT*this.l_arm*(-F(1)-F(2)+F(3)+F(4));
            tau_psi   = this.kQ*(-F(1)+F(2)-F(3)+F(4));

            % Unpack state
            z     = this.State(1);
            vz    = this.State(2);
            phi   = this.State(3);
            p     = this.State(4);
            theta = this.State(5);
            q     = this.State(6);
            psi   = this.State(7);
            r     = this.State(8);

            % Equations of motion (Euler integration, dt=0.01s)
            % Altitude: z_ddot = T/m - g (small angle approx)
            z_ddot     = T/this.m - this.g;
            vz_new     = vz  + this.dt * z_ddot;
            z_new      = z   + this.dt * vz_new;

            % Roll: phi_ddot = tau_phi/Ixx
            p_new      = p   + this.dt * (tau_phi   / this.Ixx);
            phi_new    = phi + this.dt * p_new;
            phi_new    = atan2(sin(phi_new), cos(phi_new));

            % Pitch: theta_ddot = tau_theta/Iyy
            q_new      = q   + this.dt * (tau_theta / this.Iyy);
            theta_new  = theta + this.dt * q_new;
            theta_new  = atan2(sin(theta_new), cos(theta_new));

            % Yaw: psi_ddot = tau_psi/Izz
            r_new      = r   + this.dt * (tau_psi   / this.Izz);
            psi_new    = psi + this.dt * r_new;
            psi_new    = atan2(sin(psi_new), cos(psi_new));

            % Pack new state
            this.State = [z_new;vz_new;phi_new;p_new;theta_new;q_new;psi_new;r_new];
            this.StepCount = this.StepCount + 1;

            % Build observation
            z_err = z_new - this.z_target;
            Obs   = [z_err; vz_new; phi_new; p_new; theta_new; q_new; psi_new; r_new; z_new];

            % Reward
            Reward = -this.w_z     * z_err^2 ...
                     -this.w_vz    * vz_new^2 ...
                     -this.w_angle * (phi_new^2 + theta_new^2) ...
                     -this.w_action* (a' * a);
            if abs(z_err) < 0.1
                Reward = Reward + 2.0;
            elseif abs(z_err) < 0.3
                Reward = Reward + 0.5;
            end

            % Termination
            IsDone = false;
            if abs(phi_new) > this.MaxTilt || abs(theta_new) > this.MaxTilt
                IsDone = true;
                Reward = Reward - 100;
            elseif z_new < this.MinAltitude
                IsDone = true;
                Reward = Reward - 100;
            elseif this.StepCount >= this.MaxSteps
                IsDone = true;
            end
        end

        %--- RESET ---
        function InitObs = reset(this)
            this.StepCount = 0;

            % Random initial state near hover
            z0     = this.z_target + (rand-0.5)*1.0;
            vz0    = (rand-0.5)*0.4;
            phi0   = deg2rad((rand-0.5)*20);
            p0     = (rand-0.5)*0.2;
            theta0 = deg2rad((rand-0.5)*20);
            q0     = (rand-0.5)*0.2;
            psi0   = deg2rad((rand-0.5)*20);
            r0     = (rand-0.5)*0.2;

            this.State = [z0;vz0;phi0;p0;theta0;q0;psi0;r0];

            % Random fault injection
            % 50% healthy, 50% one random motor with LoE in [0.3,1.0]
            if rand > 0.5
                fm = randi(4);
                this.Lambda = ones(4,1);
                this.Lambda(fm) = 0.3 + rand*0.7;
            else
                this.Lambda = ones(4,1);
            end

            z_err  = this.State(1) - this.z_target;
            InitObs = [z_err; this.State(2:8); this.State(1)];
        end
    end
end
