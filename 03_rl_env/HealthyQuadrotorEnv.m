classdef HealthyQuadrotorEnv < QuadrotorEnv
% Subclass that disables fault injection during training
% Used for Stage 1 curriculum training
    methods
        function this = HealthyQuadrotorEnv()
            this = this@QuadrotorEnv();
            this.w_z      = 5.0;
            this.w_vz     = 1.0;
            this.w_angle  = 0.5;
            this.w_action = 0.01;
        end
        function InitObs = reset(this)
            InitObs = reset@QuadrotorEnv(this);
            % Override: always healthy, no fault
            this.Lambda = ones(4,1);
        end
    end
end
