clc; clear; close all;

folder = "C:\Users\Edd\Desktop\track\wired";   % 改成你的文件夹
files = dir(fullfile(folder,"*.mat"));
[~,idx] = sort({files.name});
files = files(idx);

n = numel(files);
cols = ceil(sqrt(n));
rows = ceil(n/cols);

figure('Color','w');
tiledlayout(rows, cols, 'Padding','compact', 'TileSpacing','compact');

for i = 1:n
    d = load(fullfile(folder, files(i).name));
    nexttile;

    title(strrep(files(i).name,'_','\_'), ...
        'Interpreter','none','FontSize',8);

    hold on; grid on;

    % --- centerline ---
    C = d.fined_center_line;
    x = C(1,:); y = C(2,:);
    plot(x,y,'k-','LineWidth',1.2);

    % --- left boundary ---
    if isfield(d,'fined_left_bound')
        Lb = d.fined_left_bound;
        plot(Lb(1,:), Lb(2,:), 'r-','LineWidth',0.8);
    end

    % --- right boundary ---
    if isfield(d,'fined_right_bound')
        Rb = d.fined_right_bound;
        plot(Rb(1,:), Rb(2,:), 'b-','LineWidth',0.8);
    end

    % --- start point ---
    plot(x(1),y(1),'ro','MarkerSize',4,'LineWidth',1);

    axis equal;
    axis tight;
end

sgtitle(sprintf("Tracks Preview (Center + Boundaries) — %d files", n), ...
    'Interpreter','none');
