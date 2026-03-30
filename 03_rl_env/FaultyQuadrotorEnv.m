classdef FaultyQuadrotorEnv < QuadrotorEnv
% Subclass that always injects a fault during training
% Used for Stage 2 curriculum training
    methods
        function this = FaultyQuadrotorEnv()
            this = this@QuadrotorEnv();
            this.w_z      = 5.0;
            this.w_vz     = 1.0;
            this.w_angle  = 0.5;
            this.w_action = 0.01;
        end
        function InitObs = reset(this)
            InitObs = reset@QuadrotorEnv(this);
            % Override: always inject fault
            fm = randi(4);
            this.Lambda = ones(4,1);
            this.Lambda(fm) = 0.3 + rand*0.5;
        end
    end
end
