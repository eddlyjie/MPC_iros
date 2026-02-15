clc; clear; close all;

out_folder = "C:\Users\Edd\Desktop\track\wired";
if ~exist(out_folder,'dir'), mkdir(out_folder); end

ds = 1;                 % 1m per point
block_interval = 5;     % meters
block_half_length = 8;  % meters

% 10条开放赛道：每条一个“指令序列” + 宽度（可常数或分段变化）
% 指令格式：
%   {"S", L}         直行 L 米
%   {"L", R, ang}    左转：半径R，转角ang(弧度)
%   {"R", R, ang}    右转：半径R，转角ang(弧度)
%
% 说明：开放赛道不做闭合，不做回连段。

tracks = {
% name,                         width_mode, width_params,  sequence
"open_01_long_straight_W10",    "const",    10,           { {"S", 2000} };

"open_02_long_straight_W18",    "const",    18,           { {"S", 2000} };

"open_03_gentle_S_W12",         "const",    12,           { ...
    {"S", 300}, {"L", 120, pi/6}, {"S", 200}, {"R", 120, pi/3}, {"S", 200}, {"L", 120, pi/6}, {"S", 600} ...
};

"open_04_sharper_S_W10",        "const",    10,           { ...
    {"S", 250}, {"L", 90, pi/5}, {"S", 120}, {"R", 70, pi/2}, {"S", 120}, {"L", 90, pi/5}, {"S", 550} ...
};

"open_05_hairpin_W12",          "const",    12,           { ...
    {"S", 500}, {"L", 60, pi/2}, {"S", 180}, {"L", 25, pi}, {"S", 180}, {"L", 60, pi/2}, {"S", 500} ...
};

"open_06_double_hairpin_W10",   "const",    10,           { ...
    {"S", 350}, {"L", 50, pi/2}, {"S", 160}, {"L", 22, pi}, {"S", 160}, {"R", 50, pi/2}, {"S", 250}, ...
    {"R", 50, pi/2}, {"S", 160}, {"R", 22, pi}, {"S", 160}, {"L", 50, pi/2}, {"S", 350} ...
};

"open_07_chicane_W12",          "const",    12,           { ...
    {"S", 350}, {"L", 45, pi/4}, {"S", 80}, {"R", 35, pi/2}, {"S", 80}, {"L", 45, pi/4}, {"S", 700} ...
};

"open_08_fast_corners_W16",     "const",    16,           { ...
    {"S", 600}, {"L", 180, pi/3}, {"S", 300}, {"R", 160, pi/3}, {"S", 600} ...
};

"open_09_curve_width_step",     "step",     struct("W1",14,"W2",9,"s_step",800), { ...
    {"S", 300}, {"L", 120, pi/2}, {"S", 1000} ...
};

"open_10_curve_width_ramp",     "ramp",     struct("W1",18,"W2",8,"s1",400,"s2",1000), { ...
    {"S", 250}, {"L", 100, pi/2}, {"S", 1100} ...
};
};

for k = 1:size(tracks,1)
    name = tracks{k,1};
    wmode = tracks{k,2};
    wparam = tracks{k,3};
    seq = tracks{k,4};

    % 1) build centerline (raw, already ~1m step but arcs discretized)
    [x_raw, y_raw, yaw_raw] = build_path_from_sequence(seq, ds);

    % 2) resample strictly by arc-length to enforce exactly 1m spacing
    [x, y, s] = resample_open_xy_by_s(x_raw, y_raw, ds);

    % yaw from tangent
    dx = gradient(x); dy = gradient(y);
    yaw = unwrap(atan2(dy, dx));

    % 3) width along s
    W = width_profile(s, wmode, wparam);

    % 4) boundaries
    nx = -sin(yaw); ny = cos(yaw);
    xl = x + (W/2).*nx; yl = y + (W/2).*ny;
    xr = x - (W/2).*nx; yr = y - (W/2).*ny;

    % 5) shift start to origin
    x0 = x(1); y0 = y(1);
    x = x - x0; y = y - y0;
    xl = xl - x0; yl = yl - y0;
    xr = xr - x0; yr = yr - y0;

    % 6) blocks (5×M)
    Ltot = s(end);
    M = floor(Ltot / block_interval);

    bx = zeros(1,M); by = zeros(1,M); byaw = zeros(1,M);
    bhl = block_half_length * ones(1,M);
    bw  = zeros(1,M);

    for i = 1:M
        target_s = (i-1)*block_interval;
        idxs = round(target_s/ds) + 1;
        idxs = max(1, min(numel(s), idxs));
        bx(i) = x(idxs);
        by(i) = y(idxs);
        byaw(i) = yaw(idxs);
        bw(i) = W(idxs);
    end

    block_info = [bx; by; byaw; bhl; bw];  % 5×M

    % 7) start pose
    cur_pose = [0 0];     % after shifting
    cur_yaw  = yaw(1);    % rad

    % 8) export (names/shape align to your data)
    fined_center_line = [x; y];
    fined_left_bound  = [xl; yl];
    fined_right_bound = [xr; yr];
    fined_arc_length  = s;
    fined_yawang      = yaw;

    save(fullfile(out_folder, name + ".mat"), ...
        'block_info','cur_pose','cur_yaw', ...
        'fined_arc_length','fined_center_line','fined_left_bound','fined_right_bound','fined_yawang');

    fprintf("Saved %-28s  L=%.1fm  N=%d  M=%d\n", name, Ltot, numel(s), M);
end

disp("Done: 10 open tracks generated (no closure, no self-cross, no circles/ovals).");

%% ======= helpers =======
function [x,y,yaw] = build_path_from_sequence(seq, ds)
    x = 0; y = 0; yaw = 0;

    for k = 1:numel(seq)
        cmd = seq{k};

        if cmd{1} == "S"
            L = cmd{2};
            n = max(1, ceil(L/ds));
            for i = 1:n
                x(end+1) = x(end) + cos(yaw(end))*ds;
                y(end+1) = y(end) + sin(yaw(end))*ds;
                yaw(end+1) = yaw(end);
            end

        elseif cmd{1} == "L" || cmd{1} == "R"
            R = cmd{2};
            ang = cmd{3};
            dir = (cmd{1}=="L")*1 + (cmd{1}=="R")*(-1);

            L = R*ang;
            n = max(1, ceil(L/ds));
            dpsi = dir * ang / n;

            cx = x(end) - dir*R*sin(yaw(end));
            cy = y(end) + dir*R*cos(yaw(end));
            phi0 = atan2(y(end)-cy, x(end)-cx);

            for i = 1:n
                phi = phi0 + dir*(ang*i/n);
                x(end+1) = cx + R*cos(phi);
                y(end+1) = cy + R*sin(phi);
                yaw(end+1) = yaw(end) + dpsi;
            end
        else
            error("Unknown command");
        end
    end

    x = x(:)'; y = y(:)'; yaw = yaw(:)';
end

function [xr, yr, sr] = resample_open_xy_by_s(x,y,ds)
    x = x(:)'; y = y(:)';
    d = hypot(diff(x), diff(y));
    s = [0, cumsum(d)];
    L = s(end);

    sr = 0:ds:L;
    xr = interp1(s, x, sr, 'linear');
    yr = interp1(s, y, sr, 'linear');
end

function W = width_profile(s, mode, param)
    s = s(:)';

    switch mode
        case "const"
            W = param * ones(size(s));

        case "step"
            % param: W1, W2, s_step
            W = param.W1 * ones(size(s));
            W(s >= param.s_step) = param.W2;

        case "ramp"
            % param: W1 -> W2 from s1 to s2 (linear)
            W = param.W1 * ones(size(s));
            idx = (s >= param.s1) & (s <= param.s2);
            alpha = (s(idx) - param.s1) / (param.s2 - param.s1);
            W(idx) = param.W1 + (param.W2 - param.W1) * alpha;
            W(s > param.s2) = param.W2;

        otherwise
            error("Unknown width mode");
    end
end
