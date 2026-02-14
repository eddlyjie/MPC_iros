function h = plot_unit_circle()
% plot_unit_circle: Plots a black dashed unit circle on the current axes.
% This is used as the boundary for the normalized friction plot.
%
% Outputs:
%   h: Handle to the plotted line object

theta = linspace(0, 2*pi, 100);
xc = cos(theta);
yc = sin(theta);
h = plot(xc, yc, 'k--', 'LineWidth', 1.0); % Thinner dashed line

end

