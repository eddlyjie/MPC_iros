clc; clear;

files = dir('circle_track_*.mat');
if isempty(files)
    error('No circle_track_*.mat files found in current folder.');
end

for k = 1:numel(files)
    filename = files(k).name;
    fprintf('Processing %s ... ', filename);

    data = load(filename);

    if ~isfield(data, 'block_info')
        fprintf('SKIP (no block_info)\n');
        continue;
    end

    bi = data.block_info;

    % 自动判断维度：block_info 可能是 N×5 或 5×N
    if size(bi, 2) == 5
        % N×5：第5列是 width/whatever
        bi(:, 5) = bi(:, 5) / 2;
        modeStr = 'Nx5 -> divide col 5';
    elseif size(bi, 1) == 5
        % 5×N：第5行是 width/whatever
        bi(5, :) = bi(5, :) / 2;
        modeStr = '5xN -> divide row 5';
    else
        fprintf('SKIP (block_info size = %dx%d)\n', size(bi,1), size(bi,2));
        continue;
    end

    % 关键：写回 struct，再整体覆盖保存
    data.block_info = bi;
    save(filename, '-struct', 'data');

    fprintf('OK (%s)\n', modeStr);
end

fprintf('\nDone.\n');
