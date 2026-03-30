% quad_params.m
% Quadrotor physical parameters
% Source: quadcopter_parameters.m (MathWorks built-in) + extensions

% From MathWorks quadcopter_parameters.m (unchanged)
m   = 1.2722;       % total mass [kg]
g   = 9.81;         % gravity [m/s^2]
rho = 1.225;        % air density [kg/m^3]

% Propeller (from MathWorks quadcopter_parameters.m)
prop_thrust = 0.1072;   % thrust per propeller at full speed [N]
prop_power  = 0.05;     % power per propeller [W]
prop_d      = 0.254;    % propeller diameter [m]

% Derived geometry
l   = 0.225;        % arm length centre to motor [m]

% Moments of inertia (symmetric quadrotor)
Ixx = 0.0121;       % [kg.m^2] roll axis
Iyy = 0.0121;       % [kg.m^2] pitch axis
Izz = 0.0221;       % [kg.m^2] yaw axis

% Thrust and torque coefficients
% F_i = kT * omega_i^2
% Q_i = kQ * omega_i^2
kT  = 2.980e-6;     % thrust coeff [N/(rad/s)^2]
kQ  = 1.140e-7;     % torque coeff [N.m/(rad/s)^2]

% Rotor speed limits
omega_min = 0;      % [rad/s]
omega_max = 1000;   % [rad/s]

% Hover rotor speed (each motor)
% At hover: 4 * kT * omega_h^2 = m*g
% => omega_h = sqrt(m*g / (4*kT))
omega_h = sqrt(m*g / (4*kT));   % [rad/s]

% Fault model: Loss of Effectiveness per motor
% lambda=1 healthy, lambda=0 fully failed
lambda = [1; 1; 1; 1];   % [motor1;motor2;motor3;motor4]

% Simulation
dt = 0.01;          % timestep [s]
Tf = 10.0;          % episode duration [s]
