function [F_front, F_rear] = calculate_tire_forces(states, params)
% calculate_tire_forces: A MATLAB translation of the Python vehicle model's
% force calculations.
%
% Inputs:
%   states: A struct containing state vectors (v, r, ux, sa, ax)
%   params: A struct containing vehicle parameters
%
% Outputs:
%   F_front: Struct with Fx, Fy, Fz vectors for the front axle
%   F_rear:  Struct with Fx, Fy, Fz vectors for the rear axle

% Unpack vehicle parameters
la = params.la;
lb = params.lb;
hcg = params.hcg;
muf = params.muf;
mur = params.mur;
M = params.M;
g = params.g;
Caf = params.Caf;
Car = params.Car;
ratio = params.ratio;

% Unpack states (all are vectors)
v = states.v;
r = states.r;
ux = states.ux;
sa = states.sa;
ax = states.ax;

% --- Longitudinal Forces ---
% All operations are element-wise (.*, ./, .^)
axf = ax .* ratio;
axr = ax - axf;
FXF = axf .* M;
FXR = axr .* M;

% --- Vertical Forces (Load Transfer) ---
gxb = 0.0;
gzb = g;
FZR = (la * M * gzb + hcg * M .* (ax - gxb) - M * hcg .* v .* r) ./ (la + lb);
FZF = M * gzb - FZR;

% --- Tire Slip Angles ---
alphaf = atan((v + la .* r) ./ (ux + 1e-6)) - sa;
alphar = atan((v - lb .* r) ./ (ux + 1e-6));

% --- Lateral Forces (Simplified Pacejka-like model) ---
% This logic is translated directly from your Python code

% fyf_expr = 10.0 * (1 - (FXF / (muf * FZF + 1e-6))**2 - 0.03)
fyf_expr = 10.0 .* (1 - (FXF ./ (muf .* FZF + 1e-6)).^2 - 0.03);
fyr_expr = 10.0 .* (1 - (FXR ./ (mur .* FZR + 1e-6)).^2 - 0.03);

% fyf_expr_clamped = ca.fmax(-50, ca.fmin(50, fyf_expr))
% In MATLAB, max/min on vectors are different. We use element-wise.
fyf_expr_clamped = max(-50, min(50, fyf_expr));
fyr_expr_clamped = max(-50, min(50, fyr_expr));

% FYFMAX = ca.sqrt(ca.log(1 + ca.exp(fyf_expr_clamped)) / 10.0) * (muf * FZF) + 0.1
% Note: log(1 + exp(x)) is a "softplus" function
FYFMAX = sqrt(log(1 + exp(fyf_expr_clamped)) ./ 10.0) .* (muf .* FZF) + 0.1;
FYRMAX = sqrt(log(1 + exp(fyr_expr_clamped)) ./ 10.0) .* (mur .* FZR) + 0.1;

% FYF = -2 * FYFMAX * (1 / (1 + ca.exp(-(2 * Caf / (FYFMAX + 1e-6)) * alphaf)) - 0.5)
% This is a sigmoid function: 1 / (1 + exp(-x))
FYF = -2 .* FYFMAX .* (1 ./ (1 + exp(-(2 * Caf ./ (FYFMAX + 1e-6)) .* alphaf)) - 0.5);
FYR = -2 .* FYRMAX .* (1 ./ (1 + exp(-(2 * Car ./ (FYRMAX + 1e-6)) .* alphar)) - 0.5);

% --- Package Results ---
F_front.Fx = FXF;
F_front.Fy = FYF;
F_front.Fz = FZF;

F_rear.Fx = FXR;
F_rear.Fy = FYR;
F_rear.Fz = FZR;

end
